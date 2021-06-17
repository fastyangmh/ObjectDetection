# import
import torch
from torch.utils.data import Dataset
from src.project_parameters import ProjectParameters
from torchvision.datasets import VOCDetection
from typing import Optional, Callable, Tuple, Any
import numpy as np
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse

# def


def collate_fn(batch):
    images, bboxes, bboxes_length = zip(*batch)
    bboxes = np.concatenate(bboxes, 0)
    return torch.stack(images, 0), torch.tensor(bboxes), torch.tensor(bboxes_length)

# class


class VOCDetection(VOCDetection):
    def __init__(self, root: str, year: str, image_set: str, download: bool, transform: Optional[Callable]):
        super().__init__(root, year=year, image_set=image_set,
                         download=download, transform=transform)
        self.class_to_idx = {c: idx for idx, c in enumerate(sorted(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                                                    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']))}

    def _voc_to_yolo(self, anotation):
        size = anotation['annotation']['size']
        bboxes = []
        for obj in anotation['annotation']['object']:
            c = self.class_to_idx[obj['name']]
            bndbox = obj['bndbox']
            x_center = (int(bndbox['xmin']) +
                        int(bndbox['xmax'])) / (2*int(size['width']))
            y_center = (int(bndbox['ymin']) +
                        int(bndbox['ymax'])) / (2*int(size['height']))
            width = (int(bndbox['xmax']) -
                     int(bndbox['xmin']))/int(size['width'])
            height = (int(bndbox['ymax']) -
                      int(bndbox['ymin']))/int(size['height'])
            bboxes.append([c, x_center, y_center, width, height])
        bboxes = np.array(bboxes)
        return bboxes

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        anotation = self.parse_voc_xml(
            ET_parse(self.annotations[index]).getroot())
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        bboxes = self._voc_to_yolo(anotation=anotation)
        bboxes_length = len(bboxes)
        if self.transform is not None:
            bboxes = np.roll(bboxes, -1, -1)
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed['image']
            bboxes = transformed['bboxes']
            bboxes = np.roll(bboxes, 1, -1)
        image = image/255.
        return image, bboxes, bboxes_length


class YOLODataset(Dataset):
    def __init__(self) -> None:
        super().__init__()


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()
