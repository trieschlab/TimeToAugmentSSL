import torch
import os
from torch import nn
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self, args):
        super(ResNet18, self).__init__()
        self.args=args
        # regular resnet 18 as feature extractor
        resnet18 = models.resnet18(pretrained=False)
        # setting the final layer followed by an MLP to be the representation
        # network (encoder)
        setattr(resnet18, 'fc', nn.Linear(512, 256))
        self.encoder = nn.Sequential(resnet18, nn.BatchNorm1d(256),
                                     nn.ReLU(), nn.Linear(256, 128, bias=False))
        # a linear layer as projection network
        if self.args.projection and args.loss != "byol":
            self.projector = MLPHead(128, 256, 128)

    def forward(self, x):
        """
        x: image tensor of (B x 3 x 64 x 64)
        return: representation tensor h (B x FEATURE_DIM), projection tensor z
        (B x HIDDEN_DIM) that should be used by the loss function.
        """
        representation = self.encoder(x)
        if not self.args.projection or self.args.loss == "byol":
            return representation, representation
        projection = self.projector(representation)
        return representation, projection


def init_relu(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)

class ConvNet(nn.Module):
    def __init__(self, args, num_output):
        super().__init__()
        self.args=args
        self.activ_func = nn.ReLU

        drop_func = lambda : nn.Dropout(p=0.5)
        channels = 64
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            self.activ_func(),
            # batchnorm_func(args.channels),
            drop_func(),
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            self.activ_func(),
            # batchnorm_func(2 * args.channels),
            drop_func(),
            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=4, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            self.activ_func(),
            # batchnorm_func(4 * args.channels),
            drop_func(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
            # batchnorm_func(4 * args.channels),
            drop_func()
        )

        self.model.apply(init_relu)
        self.post_model= nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
        flat_features = 4 * channels

        self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
        self.train()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters:", pytorch_total_params)

    def forward(self, inputs, store=False):
        features = self.model(inputs)
        flat_features = self.post_model(features)
        representation = self.last_fc(flat_features)
        if not self.args.projection:
            return representation, representation
        projection = self.projector(representation)
        return representation, projection

class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)



def save(args, save_dir, network, optimizer):
    if args.save:
        path = os.path.join(save_dir, "model.pt")
        obj = {}
        obj["network_state_dict"] = network.state_dict()
        obj['network_optimizer_state_dict'] = optimizer.state_dict()
        torch.save(obj, path)

