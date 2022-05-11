"""
Adapted from: https://github.com/facebookresearch/swav/blob/master/src/resnet50.py
"""
import torch
from torch import nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers,
        # Supervised parameters
        num_classes,
        supervised_hidden_mlp=0,
        share_hidden_mlp=False,
        # ---
        zero_init_residual=False,
        groups=1,
        widen=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        normalize=False,
        output_dim=0,
        hidden_mlp=0,
        nmb_prototypes=0,
        eval_mode=False,
        first_conv=True,
        maxpool1=True,
        supervised_head_after_proj_head=False,
    ):
        super(ResNet, self).__init__()
        if supervised_head_after_proj_head and share_hidden_mlp:
            raise ValueError('If supervised head is after projection head, share_hidden_mlp is not allowed')
        if (output_dim == 0 or hidden_mlp == 0) and share_hidden_mlp:
            raise ValueError("there is no hidden projection head layer to share with the supervised head")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Note that the names of the norm layers are "bn1", "bn2", "bn3", regardless of which type of norm layer is used
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.supervised_head_after_proj_head = supervised_head_after_proj_head

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen

        if first_conv:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, num_out_filters, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)

        if maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        if len(layers) > 3:
            num_out_filters *= 2
            self.layer4 = self._make_layer(
                block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
            )
        else:
            self.layer4 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        self.shared_head = None
        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        elif share_hidden_mlp:
            self.shared_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp) if norm_layer is nn.BatchNorm2d else norm_layer(hidden_mlp),
                nn.ReLU(inplace=True),
            )
            self.projection_head = nn.Linear(hidden_mlp, output_dim)
            self.supervised_head = nn.Linear(hidden_mlp, num_classes)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp) if norm_layer is nn.BatchNorm2d else norm_layer(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        if supervised_head_after_proj_head:
            # supervised head (_after_ projection head and normalization)
            if supervised_hidden_mlp == 0:
                self.supervised_head = nn.Linear(output_dim, num_classes)
            else:
                self.supervised_head = nn.Sequential(
                    nn.Linear(output_dim, supervised_hidden_mlp),
                    nn.BatchNorm1d(supervised_hidden_mlp) if norm_layer is nn.BatchNorm2d else norm_layer(supervised_hidden_mlp),
                    nn.ReLU(inplace=True),
                    nn.Linear(supervised_hidden_mlp, num_classes),
                )
        elif not share_hidden_mlp:
                # supervised head
                if supervised_hidden_mlp == 0:
                    self.supervised_head = nn.Linear(num_out_filters * block.expansion, num_classes)
                else:
                    self.supervised_head = nn.Sequential(
                        nn.Linear(num_out_filters * block.expansion, supervised_hidden_mlp),
                        nn.BatchNorm1d(supervised_hidden_mlp) if norm_layer is nn.BatchNorm2d else norm_layer(supervised_hidden_mlp),
                        nn.ReLU(inplace=True),
                        nn.Linear(supervised_hidden_mlp, num_classes),
                    )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward_backbone(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4:
            x = self.layer4(x)

        if self.eval_mode:
            return x

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_proj_head(self, x):
        if self.shared_head is not None:
            x = self.shared_head(x)
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward_supervised(self, x):
        if self.shared_head is not None:
            x = self.shared_head(x)
        elif self.supervised_head_after_proj_head:
            x = self.forward_proj_head(x)
        if isinstance(x, tuple):
            x = x[0]
        return self.supervised_head(x)

    def forward_swav(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1], 0
        )
        start_idx = 0
        for end_idx in idx_crops:
            _out = torch.cat(inputs[start_idx:end_idx])

            if 'cuda' in str(self.conv1.weight.device):
                _out = self.forward_backbone(_out.cuda(non_blocking=True))
            else:
                _out = self.forward_backbone(_out)

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_proj_head(output)

    def forward(self, inputs):
        """Passes the inputs through the backbone and the heads.

        Args:
            inputs (list): List of tensor batches. The last element of the list is assumed to be the batch to pass through the supervised head,
                the remaining elements are assumed to be the multi-crop inputs as in the original version of SwAV

        Returns:
            torch.Tensor: If `inputs` contains only a single element, only a single tensor containing the output of the supervised head,
                otherwise a tuple: the output of the supervised head and the output of forward_proj_head()
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        if 'cuda' in str(self.conv1.weight.device):
            sup_out = self.forward_backbone(inputs[-1].cuda(non_blocking=True))
        else:
            sup_out = self.forward_backbone(inputs[-1])

        if len(inputs) > 1:
            swav_out = self.forward_swav(inputs[:-1])
            return self.forward_supervised(sup_out), swav_out
        return self.forward_supervised(sup_out)



class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i), nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


def resnet18(num_classes, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, **kwargs)


def resnet50(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnet50w2(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, widen=2, **kwargs)


def resnet50w4(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, widen=4, **kwargs)


def resnet50w5(num_classes, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, widen=5, **kwargs)
