import math
from typing import Type, Union, Optional, Callable, List

import torch
from torchvision.models.resnet import ResNet, conv1x1, conv3x3, Bottleneck, BasicBlock
from torchvision import models
from torch import nn, Tensor
from torch.nn import functional as F, ConvTranspose2d

from models.networks import MLPHead, ConvNet128, ConvNet128_2, ConvNet128_3, ConvNet128_3p, ResizeConv2d, \
    DecConvNet128_3


class BasicBlockDec(nn.Module):

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,padding=1,groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,groups=1, bias=False, dilation=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ConvTranspose2d(in_planes, planes, kernel_size=3, stride=stride, output_padding=1,padding=1, groups=1, bias=False, dilation=1)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ConvTranspose2d(in_planes, planes, kernel_size=1, stride=stride,output_padding=1, groups=1, bias=False, dilation=1),
                nn.BatchNorm2d(planes)
            )
class ResNet18Dec(nn.Module):

    def __init__(self, args, num_Blocks=[2,2,2,2]):
        super().__init__()
        self.in_planes = 512
        self.args =args

        self.repr = 512 if not args.linear_repr else args.num_latents

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ConvTranspose2d(64, 3 if not self.args.binocular else 6 , kernel_size=7, stride=2, padding=3, output_padding=1, bias=False)
        if args.projection:
            self.mlp = MLPHead(args, args.num_latents, args.neurons_rep, 512,batchnorm=True)
        else:
            self.mlp = nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def normalize_inputs(self,inputs):
        if self.args.inverse or self.args.predictor:
            end = F.layer_norm(inputs[:,-self.args.pred_detach:],self.args.pred_detach)
            if self.args.depth_apart:
                middle = F.layer_norm(inputs[:, -2*self.args.pred_detach:-self.args.pred_detach], (self.args.pred_detach,))
                beginning = F.layer_norm(inputs[:, :-2*self.args.pred_detach], (self.args.num_latents - 2*self.args.pred_detach,))
                return torch.cat((beginning, middle, end),dim=1)
            beginning = F.layer_norm(inputs[:, :- self.args.pred_detach], (self.args.num_latents - self.args.pred_detach,))
            return torch.cat((beginning, end),dim=1)
        return F.layer_norm(inputs,self.args.num_latents)


    def forward(self, inputs):
        if self.args.pre_norm == 2:
            inputs = self.normalize_inputs(inputs)
        x = self.mlp(inputs)
        x = x.view(inputs.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x,scale_factor=2)
        x = self.conv1(x)

        x = torch.sigmoid(x)
        x = x.view(x.size(0), 3 if not self.args.binocular else 6 , 128, 128)
        return x


class ResNet18(nn.Module):
    def __init__(self, args,small=False, number=18):
        super(ResNet18, self).__init__()
        self.args=args
        self.repr = 512 if not args.linear_repr else args.num_latents
        # regular resnet 18 as feature extractor
        if self.args.resnet_norm == "batchnorm":
            norm_layer=nn.BatchNorm2d
        elif self.args.resnet_norm == "layernorm":
            norm_layer= lambda inplanes: nn.GroupNorm(1,inplanes)
        elif self.args.resnet_norm == "dropout":
            norm_layer= lambda inplanes: nn.Dropout(p=0.5)
        elif self.args.resnet_norm == "identity":
            norm_layer= lambda inplanes: nn.Identity()

        if number == 18:
            resnet = models.resnet18(pretrained=False, norm_layer=norm_layer)
        elif number == 50:
            resnet = models.resnet50(pretrained=False, norm_layer=norm_layer)

        # setting the final layer followed by an MLP to be the representation
        # network (encoder)
        if args.binocular:
            setattr(resnet, 'conv1', nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,bias=False))
            # setattr(resnet18, 'bn1', resnet18.norm_layer(resnet18.self.inplanes))
        if small:
            resnet.conv1 = nn.Conv2d(3 if not args.binocular else 6, 64, 3, 1, 1, bias=False)
            resnet.maxpool = nn.Identity()


        if args.linear_repr:
            setattr(resnet, 'fc', nn.Linear(512, args.neurons_rep if args.extended else args.num_latents))
        else:
            setattr(resnet, 'fc', nn.Identity())

        if args.extended:
            self.encoder = nn.Sequential(resnet, nn.BatchNorm1d(args.neurons_rep),
                                         nn.ReLU(), nn.Linear(args.neurons_rep, args.num_latents, bias=False))
        else:
            self.encoder = nn.Sequential(resnet, nn.Identity())        # a linear layer as projection network
        if self.args.projection:
            self.projector = MLPHead(args, self.repr, args.neurons_rep, args.num_latents,batchnorm=self.args.proj_resnet_norm == "batchnorm"
                                     , layernorm=self.args.proj_resnet_norm == "layernorm", dropout=self.args.proj_resnet_norm == "dropout")

    def forward(self, x):
        """
        x: image tensor of (B x 3 x 64 x 64)
        return: representation tensor h (B x FEATURE_DIM), projection tensor z
        (B x HIDDEN_DIM) that should be used by the loss function.
        """
        representation = self.encoder(x)
        if not self.args.projection: #or self.args.method == "byol":
            return representation, representation
        projection = self.projector(representation)
        return representation, projection

class SmallResNet18(nn.Module):
    def __init__(self, args, small=False):
        super(SmallResNet18, self).__init__()
        self.args=args
        self.repr = 8*self.args.channels if not args.linear_repr else args.num_latents
        if self.args.resnet_norm == "batchnorm":
            norm_layer=nn.BatchNorm2d
        elif self.args.resnet_norm == "layernorm":
            norm_layer= lambda inplanes: nn.GroupNorm(1,inplanes)
        elif self.args.resnet_norm == "dropout":
            norm_layer= lambda inplanes: nn.Dropout(p=0.5)
        # regular resnet 18 as feature extractor
        resnet18 = SmallResNet(args, BasicBlock, [2, 2, 2, 2], False, True, norm_layer=norm_layer)

        if args.extended:
            self.encoder = nn.Sequential(resnet18, nn.BatchNorm1d(args.neurons_rep),
                                         nn.ReLU(), nn.Linear(args.neurons_rep, args.num_latents, bias=False))
        else:
            self.encoder = nn.Sequential(resnet18, nn.Identity())        # a linear layer as projection network
        if self.args.projection and self.args.method != "byol":
            self.projector = MLPHead(args,self.repr, args.neurons_rep, args.num_latents,batchnorm=self.args.resnet_norm == "batchnorm"
                                     ,layernorm=self.args.resnet_norm == "layernorm", dropout=self.args.resnet_norm == "dropout")
    def forward(self, x):
        """
        x: image tensor of (B x 3 x 64 x 64)
        return: representation tensor h (B x FEATURE_DIM), projection tensor z
        (B x HIDDEN_DIM) that should be used by the loss function.
        """
        representation = self.encoder(x)
        if not self.args.projection:# or self.args.method == "byol":
            return representation, representation
        projection = self.projector(representation)
        return representation, projection


class SmallResNet(nn.Module):

    def __init__(
            self,
            args,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(SmallResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.args=args
        self.inplanes = self.args.channels
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3 if not self.args.binocular else 6, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.args.channels, layers[0])
        self.layer2 = self._make_layer(block, 2*self.args.channels, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 4*self.args.channels, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 8*self.args.channels, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(8*self.args.channels * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def get_network(args, architecture=None):
    architecture = architecture if architecture is not None else args.architecture
    if architecture == "convnet":
        embed_network = ConvNet128(args, args.num_latents)
    elif architecture == "convnet_2":
        embed_network = ConvNet128_2(args, args.num_latents)
    elif architecture == "convnet_3":
        embed_network = ConvNet128_3(args, args.num_latents)
    elif architecture == "convnet_3p":
        embed_network = ConvNet128_3p(args, args.num_latents)
    elif architecture == "convnet_4":
        embed_network = ConvNet128_4(args, args.num_latents)
    elif architecture == "resnet18":
        embed_network = ResNet18(args)
    elif architecture == "resnet50":
        embed_network = ResNet18(args, number=50)
    elif architecture == "resnet18_2":
        embed_network = ResNet18(args,small=True)
    elif architecture == "smallresnet18":
        embed_network = SmallResNet18(args,small=True)
    return embed_network.to(args.device)

def get_network_decoder(args, architecture=None):
    architecture = architecture if architecture is not None else args.architecture
    if architecture == "convnet_3":
        embed_network = DecConvNet128_3(args, args.num_latents)
    elif architecture == "resnet18":
        embed_network = ResNet18Dec(args)
    return embed_network.to(args.device)