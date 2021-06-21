# import
from torch.utils.data.dataloader import DataLoader
from src.project_parameters import ProjectParameters
from src.model import create_model
from src.utils import get_transform_from_file, get_bboxes_from_anchors, get_img_with_bboxes
from PIL import Image
import torch
import numpy as np
from src.data_preparation import YOLODataset, collate_fn

# class


class Predict:
    def __init__(self, project_parameters) -> None:
        self.project_parameters = project_parameters
        self.model = create_model(project_parameters=project_parameters).eval()
        self.transform = get_transform_from_file(
            filepath=project_parameters.transform_config_path)['predict']
        self.idx_to_class = {v: k for k,
                             v in project_parameters.class_to_idx.items()}

    def get_result(self, data_path):
        result = []
        if '.png' in data_path or '.jpg' in data_path:
            image = np.array(Image.open(fp=data_path).convert('RGB'))
            image = self.transform(image=image)['image'][None, :]
            image = image/255.
            with torch.no_grad():
                anchors = self.model(image)
                bboxes, labels = get_bboxes_from_anchors(anchors=anchors, confidence_threshold=self.project_parameters.confidence_threshold,
                                                         iou_threshold=self.project_parameters.iou_threshold, labels_dict=self.idx_to_class)
                image_with_bboxes = get_img_with_bboxes(
                    image[0], bboxes[0].cpu(), resize=False, labels=labels[0])
                result.append(image_with_bboxes)
        else:
            dataset = YOLODataset(root=data_path, transform=self.transform,
                                  class_to_idx=self.project_parameters.class_to_idx, image_size=self.project_parameters.image_size)
            data_loader = DataLoader(
                dataset=dataset, batch_size=self.project_parameters.batch_size, pin_memory=self.project_parameters.use_cuda, num_workers=self.project_parameters.num_workers, collate_fn=collate_fn)
            with torch.no_grad():
                for images, _ in data_loader:
                    anchors = self.model(images)
                    bboxes, labels = get_bboxes_from_anchors(anchors=anchors, confidence_threshold=self.project_parameters.confidence_threshold,
                                                             iou_threshold=self.project_parameters.iou_threshold, labels_dict=self.idx_to_class)
                    for idx in range(len(images)):
                        image_with_bboxes = get_img_with_bboxes(
                            images[idx], bboxes[idx].cpu(), resize=False, labels=labels[idx])
                        result.append(image_with_bboxes)
        return np.array(result)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # predict the data path
    result = Predict(project_parameters=project_parameters).get_result(
        data_path=project_parameters.data_path)
