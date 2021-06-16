# import
from src.project_parameters import ProjectParameters
from torchvision.datasets import VOCDetection
from typing import Optional, Callable, Tuple, Any
from torch.utils.data import DataLoader
import numpy as np

# class


class VOCDetection(VOCDetection):
    def __init__(self, root: str, year: str, image_set: str, download: bool, transform: Optional[Callable]):
        super().__init__(root, year=year, image_set=image_set,
                         download=download, transform=transform)
        self.class_to_idx = {c: idx for idx, c in enumerate(sorted(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                                                    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']))}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image, anotation = super().__getitem__(index)
        size = anotation['annotation']['size']
        label = []
        for obj in anotation['annotation']['object']:
            c = self.class_to_idx[obj['name']]
            bndbox = obj['bndbox']
            x_center = (int(bndbox['xmin']) +
                        int(bndbox['xmax']))/(2*int(size['width']))
            y_center = (int(bndbox['ymin']) +
                        int(bndbox['ymax']))/(2*int(size['height']))
            width = (int(bndbox['xmax']) -
                     int(bndbox['xmin']))/int(size['width'])
            height = (int(bndbox['ymax']) -
                      int(bndbox['ymin']))/int(size['height'])
            label.append([c, x_center, y_center, width, height])
        return image, np.array(label)

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i  # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    #
    train_set = VOCDetection(root=project_parameters.data_path,
                             year='2012', image_set='train', download=False, transform=None)

    train_loader = DataLoader(dataset=train_set, batch_size=10)

    for x, y in train_loader:
        break
