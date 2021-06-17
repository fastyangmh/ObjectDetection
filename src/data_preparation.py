# import
import torch
from torch.utils.data import Dataset, DataLoader
from src.project_parameters import ProjectParameters
from torchvision.datasets import VOCDetection
from typing import Optional, Callable, Tuple, Any
import numpy as np
from PIL import Image
from xml.etree.ElementTree import parse as ET_parse
from os.path import join
from glob import glob
from src.utils import get_transform_from_file
from pytorch_lightning import LightningDataModule
import random

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
        self.images = np.array(self.images)
        # cannot modify the self.annotations,
        # but modify the self.targets will impact the self.annotations
        self.targets = np.array(self.targets)

    def _voc_to_yolo(self, annotation):
        size = annotation['annotation']['size']
        bboxes = []
        for obj in annotation['annotation']['object']:
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
        annotation = self.parse_voc_xml(
            ET_parse(self.annotations[index]).getroot())
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        bboxes = self._voc_to_yolo(annotation=annotation)
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
    def __init__(self, root, transform, class_to_idx) -> None:
        super().__init__()
        image_types = ['png', 'jpg']
        self.images = np.array(sorted(
            sum([glob(join(root, 'images/*.{}'.format(v))) for v in image_types], [])))
        self.annotations = np.array(
            sorted(glob(join(root, 'annotations/*.txt'))))
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.images)

    def _annotation_to_bboxes(self, annotation):
        bboxes = []
        with open(annotation, 'r') as f:
            for line in f.readlines():
                bboxes.append(np.array(line[:-1].split(' '), dtype=np.float32))
        bboxes = np.array(bboxes)
        return bboxes

    def __getitem__(self, index):
        annotation = self.annotations[index]
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        bboxes = self._annotation_to_bboxes(annotation=annotation)
        bboxes_length = len(bboxes)
        if self.transform is not None:
            bboxes = np.roll(bboxes, -1, -1)
            transformed = self.transform(image=image, bboxes=bboxes)
            image = transformed['image']
            bboxes = transformed['bboxes']
            bboxes = np.roll(bboxes, 1, -1)
        image = image/255.
        return image, bboxes, bboxes_length


class DataModule(LightningDataModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.transform_dict = get_transform_from_file(
            filepath=project_parameters.transform_config_path)

    def prepare_data(self):
        if self.project_parameters.predefined_dataset is None:
            self.dataset = {}
            for stage in ['train', 'val', 'test']:
                self.dataset[stage] = YOLODataset(root=join(self.project_parameters.data_path, stage),
                                                  transform=self.transform_dict[stage], class_to_idx=self.project_parameters.class_to_idx)
                if self.project_parameters.max_files is not None:
                    index = random.sample(
                        list(range(len(self.dataset[stage]))), k=self.project_parameters.max_files)
                    self.dataset[stage].images = self.dataset[stage].images[index]
                    self.dataset[stage].annotations = self.dataset[stage].annotations[index]
            assert self.dataset['train'].class_to_idx == self.project_parameters.class_to_idx, 'the class_to_idx is not the same. please check the class_to_idx of data. from YOLODataset: {} from argparse: {}'.format(
                self.dataset['train'].class_to_idx, self.project_parameters.class_to_idx)
        else:
            train_set = VOCDetection(root=project_parameters.data_path, year='2012',
                                     image_set='train', download=False, transform=self.transform_dict['train'])
            val_set = VOCDetection(root=project_parameters.data_path, year='2012',
                                   image_set='val', download=False, transform=self.transform_dict['val'])
            test_set = VOCDetection(root=project_parameters.data_path, year='2007',
                                    image_set='test', download=False, transform=self.transform_dict['test'])
            if self.project_parameters.max_files is not None:
                for dataset in [train_set, val_set, test_set]:
                    index = random.sample(
                        list(range(len(dataset))), k=self.project_parameters.max_files)
                    dataset.images = dataset.images[index]
                    dataset.targets = dataset.targets[index]
            self.dataset = {'train': train_set,
                            'val': val_set, 'test': test_set}
            assert self.dataset['train'].class_to_idx == self.project_parameters.class_to_idx, 'the class_to_idx is not the same. please check the class_to_idx of data. from VOCDetection: {} from argparse: {}'.format(
                self.dataset['train'].class_to_idx, self.project_parameters.class_to_idx)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['train'], batch_size=self.project_parameters.batch_size, shuffle=True, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['val'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset['test'], batch_size=self.project_parameters.batch_size, shuffle=False, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)

    def get_data_loaders(self):
        return {'train': self.train_dataloader(),
                'val': self.val_dataloader(),
                'test': self.test_dataloader()}


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # get data_module
    data_module = DataModule(project_parameters=project_parameters)
    data_module.prepare_data()

    # display the dataset information
    for stage in ['train', 'val', 'test']:
        print(stage, data_module.dataset[stage])

    # get data loaders
    data_loaders = data_module.get_data_loaders()

    #
    for x, y, z in data_loaders['train']:
        break
