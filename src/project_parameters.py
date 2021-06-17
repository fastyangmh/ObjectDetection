# import
import argparse
import torch
from os.path import join, abspath
from src.utils import load_yaml

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

        # data preparation
        self._parser.add_argument(
            '--batch_size', type=int, default=32, help='how many samples per batch to load.')
        self._parser.add_argument('--classes', type=self._str_to_str_list, required=True,
                                  help='the classes of data. if use a predefined dataset, please set value as None.')
        self._parser.add_argument('--val_size', type=float, default=0.1,
                                  help='the validation data size used for the predefined dataset.')
        self._parser.add_argument('--num_workers', type=int, default=torch.get_num_threads(
        ), help='how many subprocesses to use for data loading.')
        #self._parser.add_argument('--no_balance', action='store_true', default=False, help='whether to balance the data.')
        self._parser.add_argument('--transform_config_path', type=self._str_to_str,
                                  default='config/transform.yaml', help='the transform config path.')

    def _str_to_str(self, s):
        return None if s == 'None' or s == 'none' else s

    def _str_to_str_list(self, s):
        if '.txt' in s:
            content = []
            with open(s, 'r') as f:
                for c in f.readlines():
                    content.append(c[:-1])
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
            # the classes of predefined dataset will automatically get from data_preparation
            project_parameters.data_path = join(
                project_parameters.data_path, project_parameters.predefined_dataset)
        project_parameters.use_cuda = torch.cuda.is_available(
        ) and not project_parameters.no_cuda
        project_parameters.gpus = project_parameters.gpus if project_parameters.use_cuda else 0

        # data preparation
        if project_parameters.predefined_dataset is not None:
            project_parameters.classes = {c: idx for idx, c in enumerate(sorted(
                ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']))}
            project_parameters.num_classes = len(project_parameters.classes)
        else:
            project_parameters.classes = {
                c: idx for idx, c in enumerate(sorted(project_parameters.classes))}
            project_parameters.num_classes = len(project_parameters.classes)
        #project_parameters.use_balance = not project_parameters.no_balance and project_parameters.predefined_dataset is None
        if project_parameters.transform_config_path is not None:
            project_parameters.transform_config_path = abspath(
                project_parameters.transform_config_path)

        return project_parameters


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # display each parameter
    for k, v in vars(project_parameters).items():
        print('{:<12}= {}'.format(k, v))
