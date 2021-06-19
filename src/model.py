# import
import torch
from src.project_parameters import ProjectParameters
from pytorch_lightning import LightningModule
import timm
from os.path import dirname, basename
import torch.nn as nn

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


class Net(LightningModule):
    def __init__(self, project_parameters):
        super().__init__()
        self.project_parameters = project_parameters
        self.backbone_model = _get_backbone_model(
            project_parameters=project_parameters)
        base_number_of_channel = min(
            self.backbone_model.feature_info.channels())
        self.neck_block = Neck(base_number_of_channel=base_number_of_channel,
                               neck_output_channels=(4+1+project_parameters.num_classes)*3)

    def forward(self, x):
        downsample3, downsample4, downsample5 = self.backbone_model(x)
        y1, y2, y3 = self.neck_block(
            downsample5=downsample5, downsample4=downsample4, downsample3=downsample3)

        return y1, y2, y3


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    #
    model = Net(project_parameters=project_parameters)
    x = torch.rand(1, 3, 416, 416)
    for v in model(x):
        print(v.shape)
