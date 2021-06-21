# import
import argparse
import torch
from os.path import join, abspath, isfile
from src.utils import load_yaml, get_anchor_bbox
from timm import list_models
import numpy as np
from glob import glob

# class


class ProjectParameters:
    def __init__(self):
        self._parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # base
        self._parser.add_argument('--mode', type=str, choices=['train', 'predict', 'tune', 'evaluate'], required=True,
                                  help='if the mode equals train, will train the model. if the mode equals predict, will use the pre-trained model to predict. if the mode equals tune, will hyperparameter tuning the model. if the mode equals evaluate, will evaluate the model by the k-fold validation.')
        self._parser.add_argument(
            '--data_path', type=str, required=True, help='the data path.')
        self._parser.add_argument('--predefined_dataset', type=str, default=None, choices=[
                                  'VOCD'], help='the predefined dataset that provided the VOCD datasets.')
        self._parser.add_argument(
            '--random_seed', type=self._str_to_int, default=0, help='the random seed.')
        self._parser.add_argument(
            '--save_path', type=str, default='save/', help='the path which stores the checkpoint of PyTorch lightning.')
        self._parser.add_argument('--no_cuda', action='store_true', default=False,
                                  help='whether to use Cuda to train the model. if True which will train the model on CPU. if False which will train on GPU.')
        self._parser.add_argument('--gpus', type=self._str_to_int_list, default=-1,
                                  help='number of GPUs to train on (int) or which GPUs to train on (list or str) applied per node. if give -1 will use all available GPUs.')
        self._parser.add_argument(
            '--parameters_config_path', type=str, default=None, help='the parameters config path.')
        self._parser.add_argument(
            '--image_size', type=int, default=416, help='the image size. hypothesis the image is a square rectangle.')

        # data preparation
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')
        self._parser.add_argument('--classes', type=self._str_to_str_list, required=True,
                                  help='the classes of data. if use a predefined dataset, please set value as None.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        self._parser.add_argument('--transform_config_path', type=self._str_to_str,
                                  default='config/transform.yaml', help='the transform config path.')

        # model
        self._parser.add_argument('--in_chans', type=int, default=3,
                                  help='number of input channels / colors (default: 3).')
        self._parser.add_argument('--backbone_model', type=str, required=True,
                                  help='if you want to use a self-defined model, give the path of the self-defined model. otherwise, the provided backbone model is as a followed list. {}'.format(list_models()))
        self._parser.add_argument('--checkpoint_path', type=str, default=None,
                                  help='the path of the pre-trained model checkpoint.')
        self._parser.add_argument('--optimizer_config_path', type=str,
                                  default='config/optimizer.yaml', help='the optimizer config path.')

        # train
        self._parser.add_argument('--val_iter', type=self._str_to_int,
                                  default=None, help='the number of validation iteration.')
        self._parser.add_argument(
            '--lr', type=float, default=1e-3, help='the learning rate.')
        self._parser.add_argument(
            '--train_iter', type=int, default=100, help='the number of training iteration.')
        self._parser.add_argument('--lr_scheduler', type=str, default='CosineAnnealingLR', choices=[
                                  'StepLR', 'CosineAnnealingLR'], help='the lr scheduler while training model.')
        self._parser.add_argument(
            '--step_size', type=int, default=10, help='period of learning rate decay.')
        self._parser.add_argument('--gamma', type=int, default=0.1,
                                  help='multiplicative factor of learning rate decay.')
        self._parser.add_argument('--no_early_stopping', action='store_true',
                                  default=False, help='whether to use early stopping while training.')
        self._parser.add_argument('--patience', type=int, default=3,
                                  help='number of checks with no improvement after which training will be stopped.')
        self._parser.add_argument('--precision', type=int, default=32, choices=[
                                  16, 32], help='full precision (32) or half precision (16). Can be used on CPU, GPU or TPUs.')

        # predict
        self._parser.add_argument(
            '--confidence_threshold', type=float, default=0.5, help='the threshold of confidence.')
        self._parser.add_argument(
            '--iou_threshold', type=float, default=0.5, help='the threshold of iou.')

        # debug
        self._parser.add_argument(
            '--max_files', type=self._str_to_int, default=None, help='the maximum number of files for loading files.')
        self._parser.add_argument('--profiler', type=str, default=None, choices=[
            'simple', 'advanced'], help='to profile individual steps during training and assist in identifying bottlenecks.')
        self._parser.add_argument('--weights_summary', type=str, default=None, choices=[
                                  'top', 'full'], help='prints a summary of the weights when training begins.')
        self._parser.add_argument('--tune_debug', action='store_true',
                                  default=False, help='whether to use debug mode while tuning.')

    def _str_to_str(self, s):
        return None if s == 'None' or s == 'none' else s

    def _str_to_str_list(self, s):
        if '.txt' in s:
            content = []
            with open(abspath(s), 'r') as f:
                for line in f.readlines():
                    content.append(line[:-1])
            return content
        return [str(v) for v in s.split(',') if len(v) > 0]

    def _str_to_int(self, s):
        return None if s == 'None' or s == 'none' else int(s)

    def _str_to_int_list(self, s):
        return [int(v) for v in s.split(',') if len(v) > 0]

    def _get_new_dict(self, old_dict, yaml_dict):
        for k in yaml_dict.keys():
            del old_dict[k]
        return {**old_dict, **yaml_dict}

    def parse(self):
        project_parameters = self._parser.parse_args()
        if project_parameters.parameters_config_path is not None:
            project_parameters = argparse.Namespace(**self._get_new_dict(old_dict=vars(
                project_parameters), yaml_dict=load_yaml(filepath=abspath(project_parameters.parameters_config_path))))
        else:
            del project_parameters.parameters_config_path

        # base
        project_parameters.data_path = abspath(
            path=project_parameters.data_path)
        if project_parameters.predefined_dataset is not None:
            project_parameters.data_path = join(
                project_parameters.data_path, project_parameters.predefined_dataset)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda
        project_parameters.gpus = project_parameters.gpus if project_parameters.use_cuda else 0

        # data preparation
        if project_parameters.predefined_dataset is not None:
            project_parameters.anchor_boxes = np.array([[0.034, 0.05866667],
                                                        [0.06, 0.15466667],
                                                        [0.12, 0.096],
                                                        [0.13, 0.26666667],
                                                        [0.212, 0.5015015],
                                                        [0.302, 0.256],
                                                        [0.396, 0.712],
                                                        [0.640625, 0.43466667],
                                                        [0.848, 0.87]])
            project_parameters.classes = sorted(['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                                                 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'])
            project_parameters.class_to_idx = {
                c: idx for idx, c in enumerate(project_parameters.classes)}
            project_parameters.num_classes = len(project_parameters.classes)
        else:
            project_parameters.anchor_boxes = get_anchor_bbox(annotations=glob(
                join(project_parameters.data_path, 'train/annotations/*.txt')), n_clusters=9)
            project_parameters.classes = sorted(project_parameters.classes)
            project_parameters.class_to_idx = {
                c: idx for idx, c in enumerate(project_parameters.classes)}
            project_parameters.num_classes = len(project_parameters.classes)
        if project_parameters.transform_config_path is not None:
            project_parameters.transform_config_path = abspath(
                project_parameters.transform_config_path)

        # model
        project_parameters.optimizer_config_path = abspath(
            project_parameters.optimizer_config_path)
        if isfile(project_parameters.backbone_model):
            project_parameters.backbone_model = abspath(
                project_parameters.backbone_model)

        # train
        if project_parameters.val_iter is None:
            project_parameters.val_iter = project_parameters.train_iter
        project_parameters.use_early_stopping = not project_parameters.no_early_stopping
        if project_parameters.use_early_stopping:
            # because the PyTorch lightning needs to get validation loss in every training epoch.
            project_parameters.val_iter = 1

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for k, v in vars(project_parameters).items():
        print('{:<12}= {}'.format(k, v))
