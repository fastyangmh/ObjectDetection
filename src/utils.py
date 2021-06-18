# import
from ruamel.yaml import safe_load
from os.path import isfile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt

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
