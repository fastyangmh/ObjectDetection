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
            bbox_params = A.BboxParams(
                format='yolo') if stage != 'predict' else None
            transform_dict[stage] = A.Compose(
                transforms=transform_dict[stage], bbox_params=bbox_params)
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
    centroid_bboxes = centroid_bboxes[np.argsort(centroid_bboxes[:, 0])]
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
        bbox = bbox.astype(int)
        img_ = cv2.rectangle(
            img_, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
        if labels:
            text = labels[i]
            text += f" {bbox[4].item() :.2f}"

            img_ = cv2.putText(
                img_, text, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255))
    return img_

# class


class APAccumulator:
    def __init__(self):
        self.TP, self.FP, self.FN = 0, 0, 0

    def inc_good_prediction(self, value=1):
        self.TP += value

    def inc_bad_prediction(self, value=1):
        self.FP += value

    def inc_not_predicted(self, value=1):
        self.FN += value

    @property
    def precision(self):
        total_predicted = self.TP + self.FP
        if total_predicted == 0:
            total_gt = self.TP + self.FN
            if total_gt == 0:
                return 1.
            else:
                return 0.
        return float(self.TP) / total_predicted

    @property
    def recall(self):
        total_gt = self.TP + self.FN
        if total_gt == 0:
            return 1.
        return float(self.TP) / total_gt

    def __str__(self):
        str = ""
        str += "True positives : {}\n".format(self.TP)
        str += "False positives : {}\n".format(self.FP)
        str += "False Negatives : {}\n".format(self.FN)
        str += "Precision : {}\n".format(self.precision)
        str += "Recall : {}\n".format(self.recall)
        return str


class DetectionMAP:
    def __init__(self, n_class, pr_samples=11, overlap_threshold=0.5):
        """
        Running computation of average precision of n_class in a bounding box + classification task
        :param n_class:             quantity of class
        :param pr_samples:          quantification of threshold for pr curve
        :param overlap_threshold:   minimum overlap threshold
        """
        self.n_class = n_class
        self.overlap_threshold = overlap_threshold
        self.pr_scale = np.linspace(0, 1, pr_samples)
        self.total_accumulators = []
        self.reset_accumulators()

    def intersect_area(self, box_a, box_b):
        """
        Compute the area of intersection between two rectangular bounding box
        Bounding boxes use corner notation : [x1, y1, x2, y2]
        Args:
        box_a: (np.array) bounding boxes, Shape: [A,4].
        box_b: (np.array) bounding boxes, Shape: [B,4].
        Return:
        np.array intersection area, Shape: [A,B].
        """
        resized_A = box_a[:, np.newaxis, :]
        resized_B = box_b[np.newaxis, :, :]
        max_xy = np.minimum(resized_A[:, :, 2:], resized_B[:, :, 2:])
        min_xy = np.maximum(resized_A[:, :, :2], resized_B[:, :, :2])

        diff_xy = (max_xy - min_xy)
        inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(self, box_a, box_b):
        """
        Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
            box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
        Return:
            jaccard overlap: (np.array) Shape: [n_pred, n_gt]
        """
        inter = self.intersect_area(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
        area_a = area_a[:, np.newaxis]
        area_b = area_b[np.newaxis, :]
        union = area_a + area_b - inter
        return inter / union

    def reset_accumulators(self):
        """
        Reset the accumulators state
        TODO this is hard to follow... should use a better data structure
        total_accumulators : list of list of accumulators at each pr_scale for each class
        :return:
        """
        self.total_accumulators = []
        for i in range(len(self.pr_scale)):
            class_accumulators = []
            for j in range(self.n_class):
                class_accumulators.append(APAccumulator())
            self.total_accumulators.append(class_accumulators)

    def evaluate(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes):
        """
        Update the accumulator for the running mAP evaluation.
        For exemple, this can be called for each images
        :param pred_bb: (np.array)      Predicted Bounding Boxes [x1, y1, x2, y2] :     Shape [n_pred, 4]
        :param pred_classes: (np.array) Predicted Classes :                             Shape [n_pred]
        :param pred_conf: (np.array)    Predicted Confidences [0.-1.] :                 Shape [n_pred]
        :param gt_bb: (np.array)        Ground Truth Bounding Boxes [x1, y1, x2, y2] :  Shape [n_gt, 4]
        :param gt_classes: (np.array)   Ground Truth Classes :                          Shape [n_gt]
        :return:
        """

        if pred_bb.ndim == 1:
            pred_bb = np.repeat(pred_bb[:, np.newaxis], 4, axis=1)
        IoUmask = None
        if len(pred_bb) > 0:
            IoUmask = self.compute_IoU_mask(
                pred_bb, gt_bb, self.overlap_threshold)
        for accumulators, r in zip(self.total_accumulators, self.pr_scale):
            self.evaluate_(IoUmask, accumulators, pred_classes,
                           pred_conf, gt_classes, r)

    @staticmethod
    def evaluate_(IoUmask, accumulators, pred_classes, pred_conf, gt_classes, confidence_threshold):
        pred_classes = pred_classes.astype(int)
        gt_classes = gt_classes.astype(int)

        for i, acc in enumerate(accumulators):
            gt_number = np.sum(gt_classes == i)
            pred_mask = np.logical_and(
                pred_classes == i, pred_conf >= confidence_threshold)
            pred_number = np.sum(pred_mask)
            if pred_number == 0:
                acc.inc_not_predicted(gt_number)
                continue

            IoU1 = IoUmask[pred_mask, :]
            mask = IoU1[:, gt_classes == i]

            tp = DetectionMAP.compute_true_positive(mask)
            fp = pred_number - tp
            fn = gt_number - tp
            acc.inc_good_prediction(tp)
            acc.inc_not_predicted(fn)
            acc.inc_bad_prediction(fp)

    def compute_IoU_mask(self, prediction, gt, overlap_threshold):
        IoU = self.jaccard(prediction, gt)
        # for each prediction select gt with the largest IoU and ignore the others
        for i in range(len(prediction)):
            maxj = IoU[i, :].argmax()
            IoU[i, :maxj] = 0
            IoU[i, (maxj + 1):] = 0
        # make a mask of all "matched" predictions vs gt
        return IoU >= overlap_threshold

    @staticmethod
    def compute_true_positive(mask):
        # sum all gt with prediction of its class
        return np.sum(mask.any(axis=0))

    def compute_ap(self, precisions, recalls):
        """
        Compute average precision of a particular classes (cls_idx)
        :param cls:
        :return:
        """
        previous_recall = 0
        average_precision = 0
        for precision, recall in zip(precisions[::-1], recalls[::-1]):
            average_precision += precision * (recall - previous_recall)
            previous_recall = recall
        return average_precision

    def compute_precision_recall_(self, class_index, interpolated=True):
        precisions = []
        recalls = []
        for acc in self.total_accumulators:
            precisions.append(acc[class_index].precision)
            recalls.append(acc[class_index].recall)

        if interpolated:
            interpolated_precision = []
            for precision in precisions:
                last_max = 0
                if interpolated_precision:
                    last_max = max(interpolated_precision)
                interpolated_precision.append(max(precision, last_max))
            precisions = interpolated_precision
        return precisions, recalls

    def get_mean_average_precision(self, pred_bb, pred_classes, pred_conf, gt_bb, gt_classes, interpolated=True):
        self.evaluate(pred_bb=pred_bb, pred_classes=pred_classes,
                      pred_conf=pred_conf, gt_bb=gt_bb, gt_classes=gt_classes)
        mean_average_precision = []
        for idx in range(self.n_class):
            precisions, recalls = self.compute_precision_recall_(
                idx, interpolated)
            average_precision = self.compute_ap(precisions, recalls)
            mean_average_precision.append(average_precision)
        return np.mean(mean_average_precision)


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
