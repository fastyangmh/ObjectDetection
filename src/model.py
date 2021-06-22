# import
import torch
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningModule
import timm
from os.path import dirname, basename
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from src.utils import load_yaml, load_checkpoint, DetectionMAP
import torch.optim as optim
from src.utils import get_bboxes_from_anchors, xywh2xyxy

# def


def _get_backbone_model_from_file(filepath):
    import sys
    sys.path.append('{}'.format(dirname(filepath)))
    class_name = basename(filepath).split('.')[0]
    exec('from {} import {}'.format(*[class_name]*2))
    return eval('{}()'.format(class_name))


def _get_backbone_model(project_parameters):
    if project_parameters.backbone_model in timm.list_models():
        backbone_model = timm.create_model(model_name=project_parameters.backbone_model,
                                           pretrained=True, in_chans=project_parameters.in_chans, features_only=True, out_indices=list(range(-3, 0)))
    elif '.py' in project_parameters.backbone_model:
        backbone_model = _get_backbone_model_from_file(
            filepath=project_parameters.backbone_model)
    else:
        assert False, 'please check the backbone model. the backbone model: {}'.format(
            project_parameters.backbone_model)
    return backbone_model


def _get_optimizer(model_parameters, project_parameters):
    optimizer_config = load_yaml(
        filepath=project_parameters.optimizer_config_path)
    optimizer_name = list(optimizer_config.keys())[0]
    if optimizer_name in dir(optim):
        for name, value in optimizer_config.items():
            if value is None:
                optimizer = eval('optim.{}(params=model_parameters, lr={})'.format(
                    optimizer_name, project_parameters.lr))
            elif type(value) is dict:
                value = ('{},'*len(value)).format(*['{}={}'.format(a, b)
                                                    for a, b in value.items()])
                optimizer = eval('optim.{}(params=model_parameters, lr={}, {})'.format(
                    optimizer_name, project_parameters.lr, value))
            else:
                assert False, '{}: {}'.format(name, value)
        return optimizer
    else:
        assert False, 'please check the optimizer. the optimizer config: {}'.format(
            optimizer_config)


def _get_lr_scheduler(project_parameters, optimizer):
    if project_parameters.lr_scheduler == 'StepLR':
        lr_scheduler = StepLR(optimizer=optimizer,
                              step_size=project_parameters.step_size, gamma=project_parameters.gamma)
    elif project_parameters.lr_scheduler == 'CosineAnnealingLR':
        lr_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=project_parameters.step_size)
    else:
        assert False, 'please check the lr scheduler. the lr scheduler: {}'.format(
            project_parameters.lr_scheduler)
    return lr_scheduler


def create_model(project_parameters):
    model = Net(project_parameters=project_parameters)
    if project_parameters.checkpoint_path is not None:
        model = load_checkpoint(model=model, use_cuda=project_parameters.use_cuda,
                                checkpoint_path=project_parameters.checkpoint_path)
    return model

# class


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = []
        self.conv.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                   kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(nn.Mish(inplace=True))
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == 'linear':
            pass
        else:
            assert False, 'activate error !!!'
        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):
        return self.conv(x)


class SpatialPyramidPoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

    def forward(self, x):
        return torch.cat([self.maxpool1(x), self.maxpool2(x), self.maxpool3(x), x], dim=1)


class Neck(nn.Module):
    def __init__(self, base_number_of_channel, neck_output_channels):
        super().__init__()
        # conv x3
        self.conv_layer1 = self._make_layer(in_channels=[base_number_of_channel*4, base_number_of_channel*2, base_number_of_channel*4],
                                            out_channels=[
                                                base_number_of_channel*2, base_number_of_channel*4, base_number_of_channel*2],
                                            kernel_size=[1, 3, 1],
                                            stride=[1, 1, 1],
                                            activation=[
                                                'leaky', 'leaky', 'leaky'],
                                            bn=[True, True, True],
                                            bias=[False, False, False])

        # spp
        self.spp_block = SpatialPyramidPoolingBlock()

        # conv x3
        self.conv_layer2 = self._make_layer(in_channels=[base_number_of_channel*8, base_number_of_channel*2, base_number_of_channel*4],
                                            out_channels=[
                                                base_number_of_channel*2, base_number_of_channel*4, base_number_of_channel*2],
                                            kernel_size=[1, 3, 1],
                                            stride=[1, 1, 1],
                                            activation=[
                                                'leaky', 'leaky', 'leaky'],
                                            bn=[True, True, True],
                                            bias=[False, False, False])

        # conv1
        self.conv_layer3 = self._make_layer(in_channels=[base_number_of_channel*2],
                                            out_channels=[
                                                base_number_of_channel],
                                            kernel_size=[1],
                                            stride=[1],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # up
        self.upsample1 = nn.Upsample(scale_factor=2)

        # conv x1
        self.conv_layer4 = self._make_layer(in_channels=[base_number_of_channel*2],
                                            out_channels=[
                                                base_number_of_channel],
                                            kernel_size=[1],
                                            stride=[1],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # conv x5
        self.conv_layer5 = self._make_layer(in_channels=[base_number_of_channel*2, base_number_of_channel, base_number_of_channel*2, base_number_of_channel, base_number_of_channel*2],
                                            out_channels=[base_number_of_channel, base_number_of_channel*2,
                                                          base_number_of_channel, base_number_of_channel*2, base_number_of_channel],
                                            kernel_size=[1, 3, 1, 3, 1],
                                            stride=[1, 1, 1, 1, 1],
                                            activation=[
                                                'leaky', 'leaky', 'leaky', 'leaky', 'leaky'],
                                            bn=[True, True, True, True, True],
                                            bias=[False, False, False, False, False])

        # conv x1
        self.conv_layer6 = self._make_layer(in_channels=[base_number_of_channel],
                                            out_channels=[
                                                base_number_of_channel//2],
                                            kernel_size=[1],
                                            stride=[1],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # up
        self.upsample2 = nn.Upsample(scale_factor=2)

        # conv x1
        self.conv_layer7 = self._make_layer(in_channels=[base_number_of_channel],
                                            out_channels=[
                                                base_number_of_channel//2],
                                            kernel_size=[1],
                                            stride=[1],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # conv x5
        self.conv_layer8 = self._make_layer(in_channels=[base_number_of_channel, base_number_of_channel//2, base_number_of_channel, base_number_of_channel//2, base_number_of_channel],
                                            out_channels=[base_number_of_channel//2, base_number_of_channel,
                                                          base_number_of_channel//2, base_number_of_channel, base_number_of_channel//2],
                                            kernel_size=[1, 3, 1, 3, 1],
                                            stride=[1, 1, 1, 1, 1],
                                            activation=[
                                                'leaky', 'leaky', 'leaky', 'leaky', 'leaky'],
                                            bn=[True, True, True, True, True],
                                            bias=[False, False, False, False, False])

        # to yolo1 head
        self.conv_output_layer1 = self._make_layer(in_channels=[base_number_of_channel//2, base_number_of_channel],
                                                   out_channels=[
                                                       base_number_of_channel, neck_output_channels],
                                                   kernel_size=[3, 1],
                                                   stride=[1, 1],
                                                   activation=[
                                                       'leaky', 'linear'],
                                                   bn=[True, False],
                                                   bias=[False, True])

        # down
        self.downsample1 = self._make_layer(in_channels=[base_number_of_channel//2],
                                            out_channels=[
                                                base_number_of_channel],
                                            kernel_size=[3],
                                            stride=[2],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # conv x5
        self.conv_layer9 = self._make_layer(in_channels=[base_number_of_channel*2, base_number_of_channel, base_number_of_channel*2, base_number_of_channel, base_number_of_channel*2],
                                            out_channels=[base_number_of_channel, base_number_of_channel*2,
                                                          base_number_of_channel, base_number_of_channel*2, base_number_of_channel],
                                            kernel_size=[1, 3, 1, 3, 1],
                                            stride=[1, 1, 1, 1, 1],
                                            activation=[
                                                'leaky', 'leaky', 'leaky', 'leaky', 'leaky'],
                                            bn=[True, True, True, True, True],
                                            bias=[False, False, False, False, False])

        # to yolo2 head
        self.conv_output_layer2 = self._make_layer(in_channels=[base_number_of_channel, base_number_of_channel*2],
                                                   out_channels=[
                                                       base_number_of_channel*2, neck_output_channels],
                                                   kernel_size=[3, 1],
                                                   stride=[1, 1],
                                                   activation=[
                                                       'leaky', 'linear'],
                                                   bn=[True, False],
                                                   bias=[False, True])

        # down
        self.downsample2 = self._make_layer(in_channels=[base_number_of_channel],
                                            out_channels=[
                                                base_number_of_channel*2],
                                            kernel_size=[3],
                                            stride=[2],
                                            activation=['leaky'],
                                            bn=[True],
                                            bias=[False])

        # conv x5
        self.conv_layer10 = self._make_layer(in_channels=[base_number_of_channel*4, base_number_of_channel*2, base_number_of_channel*4, base_number_of_channel*2, base_number_of_channel*4],
                                             out_channels=[base_number_of_channel*2, base_number_of_channel*4,
                                                           base_number_of_channel*2, base_number_of_channel*4, base_number_of_channel*2],
                                             kernel_size=[1, 3, 1, 3, 1],
                                             stride=[1, 1, 1, 1, 1],
                                             activation=[
                                                 'leaky', 'leaky', 'leaky', 'leaky', 'leaky'],
                                             bn=[True, True, True, True, True],
                                             bias=[False, False, False, False, False])

        # to yolo3 head
        self.conv_output_layer3 = self._make_layer(in_channels=[base_number_of_channel*2, base_number_of_channel*4],
                                                   out_channels=[
                                                       base_number_of_channel*4, neck_output_channels],
                                                   kernel_size=[3, 1],
                                                   stride=[1, 1],
                                                   activation=[
                                                       'leaky', 'linear'],
                                                   bn=[True, False],
                                                   bias=[False, True])

    def _make_layer(self, in_channels, out_channels, kernel_size, stride, activation, bn, bias):
        conv_layer = []
        for idx in range(len(in_channels)):
            conv_layer.append(Conv_Bn_Activation(in_channels=in_channels[idx], out_channels=out_channels[idx],
                                                 kernel_size=kernel_size[idx], stride=stride[idx], activation=activation[idx], bn=bn[idx], bias=bias[idx]))
        return nn.Sequential(*conv_layer)

    def forward(self, downsample5, downsample4, downsample3):
        # conv x3
        x1 = self.conv_layer1(downsample5)

        # spp
        spp = self.spp_block(x1)

        # conv x3
        x2 = self.conv_layer2(spp)

        # conv x1
        x3 = self.conv_layer3(x2)

        # up
        up = self.upsample1(x3)

        # conv x1
        x4 = self.conv_layer4(downsample4)

        # cat
        x4 = torch.cat([x4, up], dim=1)

        # conv x5
        x5 = self.conv_layer5(x4)

        # conv x1
        x6 = self.conv_layer6(x5)

        # up
        up = self.upsample2(x6)

        # conv x1
        x7 = self.conv_layer7(downsample3)

        # cat
        x7 = torch.cat([x7, up], dim=1)

        # conv x5
        x8 = self.conv_layer8(x7)

        # to yolo1 head
        y1 = self.conv_output_layer1(x8)

        # down
        down = self.downsample1(x8)

        # cat
        x8 = torch.cat([down, x5], dim=1)

        # conv x5
        x9 = self.conv_layer9(x8)

        # to yolo2 head
        y2 = self.conv_output_layer2(x9)

        # down
        down = self.downsample2(x9)

        # cat
        x9 = torch.cat([down, x2], dim=1)

        # conv x5
        x10 = self.conv_layer10(x9)

        # to yolo3 head
        y3 = self.conv_output_layer3(x10)

        return y1, y2, y3


class YOLOLayer(nn.Module):
    """Detection layer taken and modified from https://github.com/eriklindernoren/PyTorch-YOLOv3"""

    def __init__(self, anchors, num_classes, img_dim=608, grid_size=None, iou_aware=False, repulsion_loss=False):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.obj_scale = 1
        self.noobj_scale = 100
        self.metrics = {}
        self.img_dim = img_dim
        if grid_size:
            self.grid_size = grid_size
            self.compute_grid_offsets(self.grid_size)
        else:
            self.grid_size = 0  # grid size

        self.iou_aware = iou_aware
        self.repulsion_loss = repulsion_loss

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size
        g = self.grid_size
        FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.stride = self.img_dim / self.grid_size
        # Calculate offsets for each grid
        self.grid_x = torch.arange(g).repeat(
            g, 1).view([1, 1, g, g]).type(FloatTensor)
        self.grid_y = torch.arange(g).repeat(
            g, 1).t().view([1, 1, g, g]).type(FloatTensor)
        self.scaled_anchors = FloatTensor(
            [(a_w / self.stride, a_h / self.stride) for a_w, a_h in self.anchors])
        self.anchor_w = self.scaled_anchors[:, 0:1].view(
            (1, self.num_anchors, 1, 1))
        self.anchor_h = self.scaled_anchors[:, 1:2].view(
            (1, self.num_anchors, 1, 1))

    def build_targets(self, pred_boxes, pred_cls, target, anchors, ignore_thres):

        ByteTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

        nB = pred_boxes.size(0)
        nA = pred_boxes.size(1)
        nC = pred_cls.size(-1)
        nG = pred_boxes.size(2)

        # Output tensors
        obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
        noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
        class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
        iou = FloatTensor(nB, nA, nG, nG).fill_(0)
        tx = FloatTensor(nB, nA, nG, nG).fill_(0)
        ty = FloatTensor(nB, nA, nG, nG).fill_(0)
        tw = FloatTensor(nB, nA, nG, nG).fill_(0)
        th = FloatTensor(nB, nA, nG, nG).fill_(0)
        tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

        target_boxes_grid = FloatTensor(nB, nA, nG, nG, 4).fill_(0)

        # If target is zero, then return
        if target.shape[0] == 0:
            tconf = obj_mask.float()
            # print(iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes_grid)
            return iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes_grid

        # 2 3 xy
        # 4 5 wh
        # Convert to position relative to box
        target_boxes = target[:, 2:6] * nG
        gxy = target_boxes[:, :2]
        gwh = target_boxes[:, 2:]

        # Get anchors with best iou
        ious = torch.stack([self.bbox_wh_iou(anchor, gwh)
                            for anchor in anchors])
        best_ious, best_n = ious.max(0)

        # Separate target values
        b, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()

        # Setting target boxes to big grid, it would be used to count loss
        target_boxes_grid[b, best_n, gj, gi] = target_boxes

        # Set masks
        obj_mask[b, best_n, gj, gi] = 1
        noobj_mask[b, best_n, gj, gi] = 0

        # Set noobj mask to zero where iou exceeds ignore threshold
        for i, anchor_ious in enumerate(ious.t()):
            noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

        # Coordinates
        tx[b, best_n, gj, gi] = gx - gx.floor()
        ty[b, best_n, gj, gi] = gy - gy.floor()

        # Width and height
        tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
        th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)

        # One-hot encoding of label (WE USE LABEL SMOOTHING)
        tcls[b, best_n, gj, gi, target_labels] = 0.9

        # Compute label correctness and iou at best anchor
        class_mask[b, best_n, gj, gi] = (
            pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
        iou[b, best_n, gj, gi] = self.bbox_iou(
            pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

        tconf = obj_mask.float()

        return iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes_grid

    def bbox_wh_iou(self, wh1, wh2):
        wh2 = wh2.t()
        w1, h1 = wh1[0], wh1[1]
        w2, h2 = wh2[0], wh2[1]
        inter_area = torch.min(w1, w2) * torch.min(h1, h2)
        union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
        return inter_area / union_area

    def bbox_iou(self, box1, box2, x1y1x2y2=True, get_areas=False):
        """
        Returns the IoU of two bounding boxes
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / \
                2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / \
                2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / \
                2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / \
                2, box2[:, 1] + box2[:, 3] / 2
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

    def smallestenclosing(self, pred_boxes, target_boxes):
        # Calculating smallest enclosing
        targetxc = target_boxes[..., 0]
        targetyc = target_boxes[..., 1]
        targetwidth = target_boxes[..., 2]
        targetheight = target_boxes[..., 3]

        predxc = pred_boxes[..., 0]
        predyc = pred_boxes[..., 1]
        predwidth = pred_boxes[..., 2]
        predheight = pred_boxes[..., 3]

        xc1 = torch.min(predxc - (predwidth/2), targetxc - (targetwidth/2))
        yc1 = torch.min(predyc - (predheight/2), targetyc - (targetheight/2))
        xc2 = torch.max(predxc + (predwidth/2), targetxc + (targetwidth/2))
        yc2 = torch.max(predyc + (predheight/2), targetyc + (targetheight/2))

        return xc1, yc1, xc2, yc2

    def xywh2xyxy(self, x):
        # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
        y = torch.zeros_like(x) if isinstance(
            x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def iou_all_to_all(self, a, b):
        # Calculates intersection over union area for each a bounding box with each b bounding box
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

        iw = torch.min(torch.unsqueeze(
            a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
        ih = torch.min(torch.unsqueeze(
            a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

        iw = torch.clamp(iw, min=0)
        ih = torch.clamp(ih, min=0)

        ua = torch.unsqueeze((a[:, 2] - a[:, 0]) *
                             (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

        ua = torch.clamp(ua, min=1e-8)

        intersection = iw * ih

        IoU = intersection / ua

        return IoU

    def smooth_ln(self, x, smooth=0.5):
        return torch.where(
            torch.le(x, smooth),
            -torch.log(1 - x),
            ((x - smooth) / (1 - smooth)) - math.log(1 - smooth)
        )

    def iog(self, ground_truth, prediction):

        inter_xmin = torch.max(ground_truth[:, 0], prediction[:, 0])
        inter_ymin = torch.max(ground_truth[:, 1], prediction[:, 1])
        inter_xmax = torch.min(ground_truth[:, 2], prediction[:, 2])
        inter_ymax = torch.min(ground_truth[:, 3], prediction[:, 3])
        Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
        Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
        I = Iw * Ih
        G = (ground_truth[:, 2] - ground_truth[:, 0]) * \
            (ground_truth[:, 3] - ground_truth[:, 1])
        return I / G

    def calculate_repullsion(self, y, y_hat):
        batch_size = y_hat.shape[0]
        RepGTS = []
        RepBoxes = []
        for bn in range(batch_size):
            # Repulsion between prediction bbox and neighboring target bbox, which are not target for this bounding box. (pred bbox <- -> 2nd/3rd/... by iou target bbox)
            pred_bboxes = self.xywh2xyxy(y_hat[bn, :, :4])
            bn_mask = y[:, 0] == bn
            gt_bboxes = self.xywh2xyxy(y[bn_mask, 2:] * 608)
            iou_anchor_to_target = self.iou_all_to_all(pred_bboxes, gt_bboxes)
            val, ind = torch.topk(iou_anchor_to_target, 2)
            second_closest_target_index = ind[:, 1]
            second_closest_target = gt_bboxes[second_closest_target_index]
            RepGT = self.smooth_ln(
                self.iog(second_closest_target, pred_bboxes)).mean()
            RepGTS.append(RepGT)

            # Repulsion between pred bbox and pred bbox, which are not refering to the same target bbox.
            have_target_mask = val[:, 0] != 0
            anchors_with_target = pred_bboxes[have_target_mask]
            iou_anchor_to_anchor = self.iou_all_to_all(
                anchors_with_target, anchors_with_target)
            other_mask = (torch.eye(iou_anchor_to_anchor.shape[0]) == 0).to(
                iou_anchor_to_anchor.device)
            different_target_mask = (
                ind[have_target_mask, 0] != ind[have_target_mask, 0].unsqueeze(1))
            iou_atoa_filtered = iou_anchor_to_anchor[other_mask &
                                                     different_target_mask]
            RepBox = self.smooth_ln(
                iou_atoa_filtered).sum()/iou_atoa_filtered.sum()
            RepBoxes.append(RepBox)
        return torch.stack(RepGTS).mean(), torch.stack(RepBoxes).mean()

    def forward(self, x: torch.Tensor, targets=None):
        # Tensors for cuda support
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        num_samples = x.size(0)
        grid_size = x.size(2)

        if self.iou_aware:
            not_class_channels = 6
        else:
            not_class_channels = 5
        prediction = (
            x.view(num_samples, self.num_anchors, self.num_classes +
                   not_class_channels, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )

        # Get outputs
        x = torch.sigmoid(prediction[..., 0])  # Center x
        y = torch.sigmoid(prediction[..., 1])  # Center y
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height
        pred_conf = torch.sigmoid(prediction[..., 4])  # Conf
        if not self.iou_aware:
            pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred
        else:
            pred_cls = torch.sigmoid(prediction[..., 5:-1])  # Cls pred
            pred_iou = torch.sigmoid(prediction[..., -1])  # IoU pred

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size or self.grid_x.is_cuda != x.is_cuda:
            self.compute_grid_offsets(grid_size, cuda=x.is_cuda)

        # Add offset and scale with anchors
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x + self.grid_x
        pred_boxes[..., 1] = y + self.grid_y
        pred_boxes[..., 2] = torch.exp(w) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(h) * self.anchor_h

        output = torch.cat(
            (
                pred_boxes.view(num_samples, -1, 4) * self.stride,
                pred_conf.view(num_samples, -1, 1),
                pred_cls.view(num_samples, -1, self.num_classes),
            ),
            -1,
        )

        # OUTPUT IS ALL BOXES WITH THEIR CONFIDENCE AND WITH CLASS
        if targets is None:
            return output, 0

        iou, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, target_boxes = self.build_targets(
            pred_boxes=pred_boxes,
            pred_cls=pred_cls,
            target=targets,
            anchors=self.scaled_anchors,
            ignore_thres=self.ignore_thres
        )

        # Diagonal length of the smallest enclosing box (is already squared)
        xc1, yc1, xc2, yc2 = self.smallestenclosing(
            pred_boxes[obj_mask], target_boxes[obj_mask])
        c = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + 1e-7

        # Euclidean distance between central points
        d = (tx[obj_mask] - x[obj_mask]) ** 2 + \
            (ty[obj_mask] - y[obj_mask]) ** 2

        rDIoU = d/c

        iou_masked = iou[obj_mask]
        v = (4 / (math.pi ** 2)) * torch.pow(
            (torch.atan(tw[obj_mask]/th[obj_mask])-torch.atan(w[obj_mask]/h[obj_mask])), 2)

        with torch.no_grad():
            S = 1 - iou_masked
            alpha = v / (S + v + 1e-7)

        if num_samples != 0:
            CIoUloss = (1 - iou_masked + rDIoU + alpha * v).sum(0)/num_samples
        else:
            CIoUloss = 0
        # print(torch.isnan(pred_conf).sum())

        loss_conf_noobj = F.binary_cross_entropy(
            pred_conf[noobj_mask], tconf[noobj_mask])

        if targets.shape[0] == 0:
            loss_conf_obj = 0.
            loss_cls = 0.
        else:
            loss_conf_obj = F.binary_cross_entropy(
                pred_conf[obj_mask], tconf[obj_mask])
            loss_cls = F.binary_cross_entropy(
                input=pred_cls[obj_mask], target=tcls[obj_mask])

        loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
        total_loss = CIoUloss + loss_cls + loss_conf

        if self.iou_aware:
            pred_iou_masked = pred_iou[obj_mask]

            # print("Pred iou", pred_iou.shape)
            # print("IOU masked", iou_masked.shape)
            # print("Pred iou", pred_iou)
            # print("IOU masked", iou_masked)
            # print("pred iou masked", pred_iou_masked.shape)
            # print("pred iou masked", pred_iou_masked)
            # print(F.binary_cross_entropy(pred_iou_masked, iou_masked.detach()))
            total_loss += F.binary_cross_entropy(
                pred_iou_masked, iou_masked.detach())

        if self.repulsion_loss:
            repgt, repbox = self.calculate_repullsion(targets, output)
            total_loss += 0.5 * repgt + 0.5 * repbox

        # print(f"C: {c}; D: {d}")
        # print(f"Confidence is object: {loss_conf_obj}, Confidence no object: {loss_conf_noobj}")
        # print(f"IoU: {iou_masked}; DIoU: {rDIoU}; alpha: {alpha}; v: {v}")
        # print(f"CIoU : {CIoUloss.item()}; Confindence: {loss_conf.item()}; Class loss should be because of label smoothing: {loss_cls.item()}")
        return output, total_loss


class Net(LightningModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.backbone_model = _get_backbone_model(
            project_parameters=project_parameters)
        self.neck_block = Neck(base_number_of_channel=min(self.backbone_model.feature_info.channels()),
                               neck_output_channels=(4+1+project_parameters.num_classes)*3)
        anchor_boxes = np.round(
            project_parameters.anchor_boxes*project_parameters.image_size).astype(int)
        self.yolo_head1 = YOLOLayer(
            anchors=anchor_boxes[:3], num_classes=project_parameters.num_classes, img_dim=project_parameters.image_size)
        self.yolo_head2 = YOLOLayer(
            anchors=anchor_boxes[3:6], num_classes=project_parameters.num_classes, img_dim=project_parameters.image_size)
        self.yolo_head3 = YOLOLayer(
            anchors=anchor_boxes[6:9], num_classes=project_parameters.num_classes, img_dim=project_parameters.image_size)

    def training_forward(self, x, y):
        downsample3, downsample4, downsample5 = self.backbone_model(x)
        h1, h2, h3 = self.neck_block(
            downsample5=downsample5, downsample4=downsample4, downsample3=downsample3)

        output1, loss1 = self.yolo_head1(h1, y)
        output2, loss2 = self.yolo_head2(h2, y)
        output3, loss3 = self.yolo_head3(h3, y)
        output = torch.cat((output1, output2, output3), dim=1)
        loss = (loss1 + loss2 + loss3)/3

        return output, loss

    def forward(self, x):
        downsample3, downsample4, downsample5 = self.backbone_model(x)
        h1, h2, h3 = self.neck_block(
            downsample5=downsample5, downsample4=downsample4, downsample3=downsample3)

        output1, _ = self.yolo_head1(h1, None)
        output2, _ = self.yolo_head2(h2, None)
        output3, _ = self.yolo_head3(h3, None)
        output = torch.cat((output1, output2, output3), dim=1)
        return output

    def get_progress_bar_dict(self):
        # don't show the loss value
        items = super().get_progress_bar_dict()
        items.pop('loss', None)
        return items

    def _parse_outputs(self, outputs):
        epoch_loss = []
        epoch_mAP = []
        for step in outputs:
            epoch_loss.append(step['loss'].item())
            epoch_mAP.append(step['mAP'])
        return epoch_loss, epoch_mAP

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.training_forward(x, y)
        bboxes, _ = get_bboxes_from_anchors(anchors=y_hat, confidence_threshold=self.project_parameters.confidence_threshold,
                                            iou_threshold=self.project_parameters.iou_threshold, labels_dict=self.project_parameters.idx_to_class)
        bboxes = torch.cat(bboxes, 0)
        pred_bb = bboxes[:, :4].cpu().data.numpy()
        pred_cls = bboxes[:, 5:].argmax(-1).cpu().data.numpy()
        pred_conf = bboxes[:, 4].cpu().data.numpy()
        gt_bb = xywh2xyxy(y[:, 2:]).cpu().data.numpy() * \
            self.project_parameters.image_size
        gt_cls = y[:, 1].cpu().data.numpy()
        mAP = DetectionMAP(self.project_parameters.num_classes).get_mean_average_precision(
            pred_bb=pred_bb, pred_classes=pred_cls, pred_conf=pred_conf, gt_bb=gt_bb, gt_classes=gt_cls)
        return {'loss': loss, 'mAP': mAP}

    def training_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_mAP = self._parse_outputs(outputs=outputs)
        self.log('training loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('training mAP', torch.tensor(epoch_mAP))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.training_forward(x, y)
        bboxes, _ = get_bboxes_from_anchors(anchors=y_hat, confidence_threshold=self.project_parameters.confidence_threshold,
                                            iou_threshold=self.project_parameters.iou_threshold, labels_dict=self.project_parameters.idx_to_class)
        bboxes = torch.cat(bboxes, 0)
        pred_bb = bboxes[:, :4].cpu().data.numpy()
        pred_cls = bboxes[:, 5:].argmax(-1).cpu().data.numpy()
        pred_conf = bboxes[:, 4].cpu().data.numpy()
        gt_bb = xywh2xyxy(y[:, 2:]).cpu().data.numpy() * \
            self.project_parameters.image_size
        gt_cls = y[:, 1].cpu().data.numpy()
        mAP = DetectionMAP(self.project_parameters.num_classes).get_mean_average_precision(
            pred_bb=pred_bb, pred_classes=pred_cls, pred_conf=pred_conf, gt_bb=gt_bb, gt_classes=gt_cls)
        return {'loss': loss, 'mAP': mAP}

    def validation_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_mAP = self._parse_outputs(outputs=outputs)
        self.log('validation loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('validation mAP', torch.tensor(epoch_mAP))

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, loss = self.training_forward(x, y)
        bboxes, _ = get_bboxes_from_anchors(anchors=y_hat, confidence_threshold=self.project_parameters.confidence_threshold,
                                            iou_threshold=self.project_parameters.iou_threshold, labels_dict=self.project_parameters.idx_to_class)
        bboxes = torch.cat(bboxes, 0)
        pred_bb = bboxes[:, :4].cpu().data.numpy()
        pred_cls = bboxes[:, 5:].argmax(-1).cpu().data.numpy()
        pred_conf = bboxes[:, 4].cpu().data.numpy()
        gt_bb = xywh2xyxy(y[:, 2:]).cpu().data.numpy() * \
            self.project_parameters.image_size
        gt_cls = y[:, 1].cpu().data.numpy()
        mAP = DetectionMAP(self.project_parameters.num_classes).get_mean_average_precision(
            pred_bb=pred_bb, pred_classes=pred_cls, pred_conf=pred_conf, gt_bb=gt_bb, gt_classes=gt_cls)
        return {'loss': loss, 'mAP': mAP}

    def test_epoch_end(self, outputs) -> None:
        epoch_loss, epoch_mAP = self._parse_outputs(outputs=outputs)
        self.log('test loss', np.mean(epoch_loss),
                 on_epoch=True, prog_bar=True)
        self.log('test mAP', torch.tensor(epoch_mAP))

    def configure_optimizers(self):
        optimizer = _get_optimizer(model_parameters=self.parameters(
        ), project_parameters=self.project_parameters)
        if self.project_parameters.step_size > 0:
            lr_scheduler = _get_lr_scheduler(
                project_parameters=self.project_parameters, optimizer=optimizer)
            return [optimizer], [lr_scheduler]
        else:
            return optimizer


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    model.summarize()

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.image_size, project_parameters.image_size)

    # get model output
    y = model(x)

    # display the dimension of input and output
    print(x.shape)
    print(y.shape)

    #

    idx_to_class = {v: k for k, v in project_parameters.class_to_idx.items()}
    bboxes, labels = get_bboxes_from_anchors(anchors=y, confidence_threshold=project_parameters.confidence_threshold,
                                             iou_threshold=project_parameters.iou_threshold, labels_dict=idx_to_class)
