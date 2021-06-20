# import
from ruamel.yaml import safe_load
from os.path import isfile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torchvision.extension import _assert_has_ops
from torch import Tensor

# global variables
BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White

# def


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content = safe_load(f)
    return content


def get_transform_from_file(filepath):
    if filepath is None:
        return {}.fromkeys(['train', 'val', 'test', 'predict'], None)
    elif isfile(filepath):
        transform_dict = {}
        transform_config = load_yaml(filepath=filepath)
        for stage in transform_config.keys():
            transform_dict[stage] = []
            if type(transform_config[stage]) != dict:
                transform_dict[stage] = None
                continue
            for name, value in transform_config[stage].items():
                if name == 'ToTensorV2':
                    transform_dict[stage].append(ToTensorV2())
                elif value is None:
                    transform_dict[stage].append(
                        eval('A.{}()'.format(name)))
                else:
                    if type(value) is dict:
                        value = ('{},'*len(value)).format(*
                                                          ['{}={}'.format(a, b) for a, b in value.items()])
                    transform_dict[stage].append(
                        eval('A.{}({})'.format(name, value)))
            transform_dict[stage] = A.Compose(transforms=transform_dict[stage],
                                              bbox_params=A.BboxParams(format='yolo'))
        return transform_dict
    else:
        assert False, 'please check the transform config path: {}'.format(
            filepath)


def load_checkpoint(model, use_cuda, checkpoint_path):
    map_location = torch.device(
        device='cuda') if use_cuda else torch.device(device='cpu')
    checkpoint = torch.load(f=checkpoint_path, map_location=map_location)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def get_anchor_bbox(annotations, n_clusters):
    bboxes = []
    for filepath in annotations:
        with open(filepath, 'r') as f:
            bboxes.append(f.readline()[:-1].split(' ')[-2:])
    bboxes = np.array(bboxes, dtype=np.float32)
    kmeans = AnchorKmeans(n_clusters=n_clusters, distance_method=np.median)
    centroid_bboxes = kmeans(bboxes=bboxes)
    return centroid_bboxes


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def bbox_iou(box1, box2, x1y1x2y2=True, get_areas=False):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = (b1_area + b2_area - inter_area + 1e-16)

    if get_areas:
        return inter_area, union_area

    iou = inter_area / union_area
    return iou


def nms_with_depth(bboxes, confidence, iou_threshold, depth_layer, depth_threshold):
    if len(bboxes) == 0:
        return bboxes

    for i in range(bboxes.shape[0]):
        for j in range(i+1, bboxes.shape[0]):
            iou = bbox_iou(bboxes[i], bboxes[j])
            if iou > iou_threshold:
                # Getting center depth points of both bboxes
                D_oi = depth_layer[(bboxes[i, 0] + bboxes[i, 2]) //
                                   2, (bboxes[i, 1] + bboxes[i, 3])//2]
                D_oj = depth_layer[(bboxes[j, 0] + bboxes[j, 2]) //
                                   2, (bboxes[j, 1] + bboxes[j, 3])//2]
                if D_oi - D_oj < depth_threshold:
                    average_depth_oi = depth_layer[bboxes[i, 0]                                                   : bboxes[i, 2], bboxes[i, 1]: bboxes[i, 3]]
                    average_depth_oj = depth_layer[bboxes[j, 0]                                                   : bboxes[j, 2], bboxes[j, 1]: bboxes[j, 3]]
                    score_oi = confidence[i] + 1/torch.log(average_depth_oi)
                    score_oj = confidence[j] + 1/torch.log(average_depth_oj)
                    if score_oi > score_oj:
                        confidence[j] = 0
                    else:
                        confidence[i] = 0

    return confidence != 0


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than iou_threshold with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices
            of the elements that have been kept
            by NMS, sorted in decreasing order of scores
    """
    _assert_has_ops()
    return torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def get_bboxes_from_anchors(anchors, confidence_threshold, iou_threshold, labels_dict, depth_layer=None, depth_threshold=0.1):
    nbatches = anchors.shape[0]
    batch_bboxes = []
    labels = []

    for nbatch in range(nbatches):
        img_anchor = anchors[nbatch]
        confidence_filter = img_anchor[:, 4] > confidence_threshold
        img_anchor = img_anchor[confidence_filter]
        if depth_layer != None:
            keep = nms_with_depth(xywh2xyxy(
                img_anchor[:, :4]), img_anchor[:, 4], iou_threshold, depth_layer, depth_threshold)
        else:
            keep = nms(xywh2xyxy(img_anchor[:, :4]),
                       img_anchor[:, 4], iou_threshold)

        img_bboxes = img_anchor[keep]
        batch_bboxes.append(img_bboxes)
        if len(img_bboxes) == 0:
            labels.append([])
            continue
        labels.append([labels_dict[x.item()]
                       for x in img_bboxes[:, 5:].argmax(1)])

    return batch_bboxes, labels


def get_img_with_bboxes(img, bboxes, resize=True, labels=None, confidences=None):
    c, h, w = img.shape

    bboxes_xyxy = bboxes.clone()
    bboxes_xyxy[:, :4] = xywh2xyxy(bboxes[:, :4])
    if resize:
        bboxes_xyxy[:, 0] *= w
        bboxes_xyxy[:, 1] *= h
        bboxes_xyxy[:, 2] *= w
        bboxes_xyxy[:, 3] *= h

        bboxes_xyxy[:, 0:4] = bboxes_xyxy[:, 0:4].round()

    arr = bboxes_xyxy.numpy()

    img = img.permute(1, 2, 0)
    img = img.numpy()
    img = (img * 255).astype(np.uint8)

    # Otherwise cv2 rectangle will return UMat without paint
    img_ = img.copy()

    for i, bbox in enumerate(arr):
        img_ = cv2.rectangle(
            img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
        if labels:
            text = labels[i]
            text += f" {bbox[4].item() :.2f}"

            img_ = cv2.putText(
                img_, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255))
    return img_

# class


class Visualize:
    def __init__(self) -> None:
        pass

    def _parse_yolo_bbox(self, img, bbox):
        height, width, _ = img.shape
        x_center, y_center, yolo_w, yolo_h = bbox
        x_center *= 2*width
        y_center *= 2*height
        yolo_w *= width
        yolo_h *= height
        x_min = int((x_center-yolo_w)/2)
        y_min = int((y_center-yolo_h)/2)
        x_max = int(x_center-x_min)
        y_max = int(y_center-y_min)
        return x_min, x_max, y_min, y_max

    def _visualize_yolo_bbox(self, img, bbox, class_name, color=BOX_COLOR, thickness=2):
        """Visualizes a single bounding box on the image"""
        x_min, x_max, y_min, y_max = self._parse_yolo_bbox(img=img, bbox=bbox)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=color, thickness=thickness)

        ((text_width, text_height), _) = cv2.getTextSize(
            class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                      (x_min + text_width, y_min), BOX_COLOR, -1)
        cv2.putText(
            img,
            text=class_name,
            org=(x_min, y_min - int(0.3 * text_height)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.35,
            color=TEXT_COLOR,
            lineType=cv2.LINE_AA,
        )
        return img

    def __call__(self, image, bboxes, category_ids, category_id_to_name):
        img = image.copy()
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = self._visualize_yolo_bbox(img, bbox, class_name)
        plt.figure(figsize=(12, 12))
        plt.axis('off')
        plt.imshow(img)
        plt.show()


class AnchorKmeans:
    def __init__(self, n_clusters, distance_method=np.median):
        self.n_clusters = n_clusters
        self.distance_method = distance_method

    def _calculate_iou(self, bbox, clusters):
        x = np.minimum(bbox[0], clusters[:, 0])
        y = np.minimum(bbox[1], clusters[:, 1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")
        intersection = x * y
        box_area = bbox[0] * bbox[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]
        return intersection / (box_area + cluster_area - intersection)

    def calculate_average_iou(self, bboxes, clusters):
        return np.mean([np.max(self._calculate_iou(bboxes[i], clusters)) for i in range(len(bboxes))])

    def __call__(self, bboxes):
        #distances is iou
        number_of_boxes = len(bboxes)
        distances = np.zeros(shape=(number_of_boxes, self.n_clusters))
        last_clusters = np.zeros(number_of_boxes)
        clusters = bboxes[random.choices(
            list(range(number_of_boxes)), k=self.n_clusters)]
        while True:
            for idx in range(number_of_boxes):
                distances[idx] = self._calculate_iou(bboxes[idx], clusters)
            nearest_clusters = np.argmax(distances, axis=1)
            if (last_clusters == nearest_clusters).all():
                break
            for cluster_idx in range(self.n_clusters):
                clusters[cluster_idx] = self.distance_method(
                    bboxes[nearest_clusters == cluster_idx], axis=0)
            last_clusters = nearest_clusters
        return clusters
