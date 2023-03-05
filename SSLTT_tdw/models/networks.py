

import torch
from torch.distributions import Normal
from torch.nn import ModuleList, ConvTranspose2d
from torch import nn
from torch.nn import functional as F

from tools.utils import size_space


def init_leaky(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.constant_(m.bias, 0.01)


# initializtion tanh
def init_tanh(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(m.bias, 0.01)


# initializtion linear
def init_lin(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.constant_(m.bias, 0.01)


# initializtion relu
def init_relu(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.01)


def init_default(m):
    if m == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.01)

class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class Actor_SAC(nn.Module):
    def __init__(self, mlp, action_space, args):
        super().__init__()
        self.mlp = mlp
        device = args.device
        self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.).to(device)
        self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.).to(device)
        self.dist = Normal
        n_out = size_space(action_space)
        self.mu_gen = nn.Linear(in_features=self.mlp.num_output, out_features=n_out)
        self.mu_gen.apply(init_lin)
        self.std_gen = nn.Linear(in_features=self.mlp.num_output, out_features=n_out)
        self.std_gen.apply(init_lin)
        # self.LOG_SIG_MAX = 2
        # self.LOG_SIG_MIN = -20

    def bound_action(self, action, action_logprob, dist):
        action2 = torch.tanh(action)
        action = action2 * self.action_scale + self.action_bias

        # https://arxiv.org/pdf/1812.05905.pdf
        action_logprob -= (torch.log(self.action_scale * (1 - action2.pow(2)) + 1e-6))
        action_logprob = action_logprob.sum(dim=1)
        return action, action_logprob.view(-1, 1), dist

    def bound_action_old(self, action, action_logprob, dist):
        action2 = torch.tanh(action)
        action = action2 * self.action_scale + self.action_bias

        # https://arxiv.org/pdf/1812.05905.pdf
        action_logprob -= (torch.log(self.action_scale * (1 - action2.pow(2)) + 1e-6)).sum(dim=1)
        return action, action_logprob.view(-1, 1), dist

    def forward(self, inputs, deterministic=False, **kwargs):
        actor_features = self.mlp(inputs, **kwargs)
        mu = self.mu_gen(actor_features)
        std = self.std_gen(actor_features)
        # scale = torch.exp(torch.clamp(std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX))
        scale = torch.exp(std)
        dist = self.dist(mu, scale)
        naction = dist.rsample() if not deterministic else mu  # torch.tanh(mu) * self.action_scale + self.action_bias
        naction_log_probs = dist.log_prob(naction)  # .sum(dim=1)
        action, action_log_probs, dist = self.bound_action(naction, naction_log_probs, dist)
        return action, action_log_probs, dist


class MLP(nn.Module):
    def __init__(self, num_inputs, num_output, hidden_size=64, activation=nn.Tanh, num_layers=1,
                 last_linear=True, dropout=0, init_zero=None, batch_norm=False, goal_size=None, pre_norm=False):
        super(MLP, self).__init__()
        assert num_layers >= 0, "cant have less than 0 hidden_ layers"
        self.num_output = num_output
        self.num_inputs = num_inputs
        if activation == "relu":
            activation = nn.ReLU
        elif activation == "leaky":
            activation = nn.LeakyReLU
        elif activation == "elu":
            activation = nn.ELU

        if activation == nn.Tanh:
            activ_fc = init_tanh
        elif activation == nn.LeakyReLU:
            activ_fc = init_leaky
        elif activation == nn.ReLU:
            activ_fc = init_relu
        else:
            activ_fc = init_default
        self.activ_fc = activ_fc
        if num_layers == 0:
            assert last_linear, "can not have 0 hidden layers and no last linear layer"
            assert init_zero is None, "can currently handle"
            self.model = nn.Sequential(nn.Linear(num_inputs + (goal_size if goal_size is not None else 0), num_output))
        else:
            if not last_linear:
                num_layers = num_layers - 1
            layers = []
            if pre_norm ==1:
                layers.append(nn.BatchNorm1d(num_inputs))
            elif pre_norm == 2:
                layers.append(nn.LayerNorm(num_inputs))
            layers.append(nn.Linear(num_inputs, hidden_size))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

            for i in range(num_layers - 1):
                layers.append(nn.Linear(hidden_size + (goal_size if goal_size is not None else 0), hidden_size))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            layers.append(nn.Linear(hidden_size + (goal_size if goal_size is not None else 0), num_output))
            if not last_linear:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(p=dropout))
            self.model = ModuleList(layers).apply(activ_fc)

            if last_linear:
                layers[-1].apply(init_lin)

        self.train()

    def get_gradient(self):
        return torch.norm(torch.stack([torch.norm(p.grad.detach(), 1) for p in self.parameters()]), 1)

    def forward(self, inputs):
        out = inputs
        for i, layer in enumerate(self.model):
            out = layer(out)
        return out

    def forward_double(self, inputs, actions):
        out = torch.cat((inputs,actions),dim=1)
        for i, layer in enumerate(self.model):
            out = layer(out)
        return out

class ConvMLP(MLP):
    def __init__(self, args, extra=0, **kwargs):
        channels = 32
        super().__init__(num_inputs=4*channels + extra, **kwargs)
        self.activ_func = nn.ReLU
        num_channels = 3 if not args.binocular else 6
        self.conv_model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            self.activ_func(),
            nn.Conv2d(in_channels=channels, out_channels=2*channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            self.activ_func(),
            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=4, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            self.activ_func(),
            nn.Conv2d(in_channels=4*channels, out_channels=4*channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
            nn.AvgPool2d(4),
            nn.Flatten()
        )
        self.conv_model.apply(init_relu)

    def forward(self, inputs):
        features = self.conv_model(inputs)
        out = super().forward(features)
        return out

    def forward_double(self, inputs, actions):
        features = self.conv_model(inputs)
        features = torch.cat((features, actions), dim=1)
        out = super().forward(features)
        return out


class DoubleConvNet128(nn.Module):
    def __init__(self, args, num_output, **kwargs):
        super().__init__()
        self.head1 = ConvNet128(args, num_output//2,**kwargs)
        self.head2 = ConvNet128(args, num_output//2,**kwargs)

    def forward(self,inputs):
        out1, projec1 = self.head1(inputs)
        out2, projec2 = self.head2(inputs)

        return torch.cat((out1, out2), dim=1), torch.cat((projec1, projec2), dim=1)


class ConvNet128(nn.Module):
    def __init__(self, args, num_output):
        super().__init__()
        self.args=args
        self.activ_func = nn.ReLU
        coef_channels = args.channels
        if args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 0:
            drop_func = nn.Identity
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=args.channels, kernel_size=8, stride=4, padding=0),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=args.channels, out_channels=2*args.channels, kernel_size=4, stride=2, padding=0),  # 31 - 4 / 2 = 13 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=2*args.channels, out_channels=4*args.channels, kernel_size=2, stride=2, padding=0),  # 14 - 2 / 2 =  6 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=4*args.channels, out_channels=4*args.channels, kernel_size=2, stride=2, padding=0),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity(),

        )

        self.model.apply(init_relu)
        if args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(3), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Flatten()
            flat_features = 9*4*args.channels


        if args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                nn.LeakyReLU(),
                # drop_func(),
                nn.Linear(args.neurons_rep, num_output)
            )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        if args.projection:
            self.projector = MLPHead(num_output, args.neurons_rep, num_output)

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


class ConvNet128_2(nn.Module):
    def __init__(self, args, num_output, dropout=True):
        super().__init__()
        self.args=args
        self.activ_func = nn.ReLU
        if args.dropout == 0 or args.dropout == 3 or not dropout:
            drop_func = nn.Identity
        elif args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)

        flat_drop_func = nn.Identity if args.dropout != 3 else (lambda : nn.Dropout(p=args.drop_val))

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=args.channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=args.channels, out_channels=2*args.channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=2*args.channels, out_channels=4*args.channels, kernel_size=2, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=4*args.channels, out_channels=4*args.channels, kernel_size=2, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity(),
        )

        self.model.apply(init_relu)
        if args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(5), nn.Flatten(), flat_drop_func())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Sequential(nn.Flatten(), flat_drop_func())
            flat_features = 25*4*args.channels

        if args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # drop_func(),
                # nn.Tanh()
                nn.LeakyReLU(),
                nn.Linear(args.neurons_rep, num_output)
            )
            # self.last_fc = nn.Sequential(
            #     nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # nn.Tanh(),
                # nn.Linear(args.neurons_rep, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # nn.Tanh(),
                # nn.Linear(args.neurons_rep, num_output)
            # )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        if args.projection:
            self.projector = MLPHead(num_output, args.neurons_rep, num_output)

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


class ConvNet128_3(nn.Module):
    def __init__(self, args, num_output, dropout=True):
        super().__init__()
        self.args=args
        if args.activation == "relu":
            self.activ_func = nn.ReLU
        if args.activation == "leaky":
            self.activ_func = nn.LeakyReLU
        if args.activation == "elu":
            self.activ_func = nn.ELU

        if args.dropout == 0 or args.dropout == 3 or not dropout:
            drop_func = nn.Identity
        elif args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)

        if not args.batchnorm:
            batchnorm_func = nn.Identity
            batchnorm_func2 = nn.Identity
        else:
            batchnorm_func = nn.BatchNorm2d
            batchnorm_func2 = nn.BatchNorm1d

        channels = 3 if not self.args.binocular else 6
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=args.channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            batchnorm_func(args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=args.channels, out_channels=2*args.channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            batchnorm_func(2 * args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=2*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            batchnorm_func(4 * args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=4*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            batchnorm_func(4 * args.channels),
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity()
        )

        self.model.apply(init_relu if args.activation == "relu" else init_leaky)
        if args.max:
            self.post_model= nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        elif args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Flatten()
            flat_features = 16*4*args.channels

        if not self.args.linear_repr:
            self.last_fc = nn.Identity()
        elif args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # nn.Sigmoid(),
                batchnorm_func2(args.neurons_rep),
                nn.ReLU(),
                drop_func() if args.ext_dropout else nn.Identity,
                nn.Linear(args.neurons_rep, num_output)
            )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        self.repr = num_output if self.args.linear_repr else flat_features
        if args.projection:
            # self.projector = MLPHead(num_output, args.neurons_rep, num_output)
            self.projector = MLP(self.repr, num_output, args.neurons_rep, args.proj_activation, args.proj_layers, last_linear=True,dropout=args.proj_dropout, batch_norm=args.batchnorm)

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


class DecConvNet128_3(nn.Module):
    def __init__(self, args, num_output, dropout=True):
        super().__init__()
        self.args=args
        if args.activation == "relu":
            self.activ_func = nn.ReLU
        if args.activation == "leaky":
            self.activ_func = nn.LeakyReLU
        if args.activation == "elu":
            self.activ_func = nn.ELU

        if args.dropout == 0 or args.dropout == 3 or not dropout:
            drop_func = nn.Identity
        elif args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)

        if not args.batchnorm:
            batchnorm_func = nn.Identity
            batchnorm_func2 = nn.Identity
        else:
            batchnorm_func = nn.BatchNorm2d
            batchnorm_func2 = nn.BatchNorm1d

        channels = 3 if not self.args.binocular else 6
        self.model = nn.Sequential(
            batchnorm_func(4 * args.channels),
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity(),
            ConvTranspose2d(4 * args.channels, 4 * args.channels, kernel_size=4, stride=2, padding=1),
            batchnorm_func(4*args.channels),
            self.activ_func(),
            drop_func(),
            ConvTranspose2d(4 * args.channels, 2*args.channels, kernel_size=4, stride=2, padding=1),
            batchnorm_func(2 * args.channels),
            self.activ_func(),
            drop_func(),
            ConvTranspose2d(2*args.channels, args.channels, kernel_size=4, stride=2, padding=1),
            batchnorm_func(args.channels),
            self.activ_func(),
            drop_func(),
            ConvTranspose2d(args.channels, channels, kernel_size=8, stride=4, padding=2),
            # nn.Conv2d(in_channels=4*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
        )

        self.model.apply(init_relu if args.activation == "relu" else init_leaky)
        if args.max:
            self.post_model= nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        elif args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Flatten()
            flat_features = 16*4*args.channels

        if not self.args.linear_repr:
            self.last_fc = nn.Identity()
        elif args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(num_output, args.neurons_rep),
                batchnorm_func2(args.neurons_rep),
                nn.ReLU(),
                drop_func() if args.ext_dropout else nn.Identity,
                nn.Linear(args.neurons_rep,flat_features)
            )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        self.repr = num_output if self.args.linear_repr else flat_features
        if args.projection:
            self.projector = MLP(self.args.num_latents, flat_features, args.neurons_rep, args.proj_activation, args.proj_layers, last_linear=True,dropout=args.proj_dropout, batch_norm=args.batchnorm)

        self.train()
        pytorch_total_params = sum(p.numel() for p in self.parameters())
        print("Number of parameters:", pytorch_total_params)

    def normalize_inputs(self,inputs):
        if self.args.inverse or self.args.predictor:
            end = F.layer_norm(inputs[:,-self.args.pred_detach:],(self.args.pred_detach,))
            if self.args.depth_apart:
                middle = F.layer_norm(inputs[:, -2*self.args.pred_detach:-self.args.pred_detach], (self.args.pred_detach,))
                beginning = F.layer_norm(inputs[:, :-2*self.args.pred_detach], (self.args.num_latents - 2*self.args.pred_detach,))
                return torch.cat((beginning, middle, end),dim=1)
            beginning = F.layer_norm(inputs[:, :- self.args.pred_detach], (self.args.num_latents - self.args.pred_detach,))
            return torch.cat((beginning, end),dim=1)
        return F.layer_norm(inputs,self.args.num_latents)


    def forward(self, inputs, store=False):
        if self.args.pre_norm == 2:
            inputs = self.normalize_inputs(inputs)

        if self.args.projection:
            inputs = self.projector(inputs)
        x = self.last_fc(inputs)
        x = x.view(x.size(0), 256, 1, 1)
        x = torch.nn.functional.interpolate(x, scale_factor=4)
        x = self.model(x)
        x = torch.sigmoid(x)
        x = x.view(x.size(0), 3 if not self.args.binocular else 6, 128, 128)
        return x

class ConvNet128_3p(nn.Module):
    def __init__(self, args, num_output, dropout=True):
        super().__init__()
        self.args=args
        if args.activation == "relu":
            self.activ_func = nn.ReLU
        if args.activation == "leaky":
            self.activ_func = nn.LeakyReLU
        if args.activation == "elu":
            self.activ_func = nn.ELU

        if args.dropout == 0 or args.dropout == 3 or not dropout:
            drop_func = nn.Identity
        elif args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)
        channels = 3 if not self.args.binocular else 6
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=args.channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=args.channels, out_channels=2*args.channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=2*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=4*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity()
        )

        self.model.apply(init_relu if args.activation == "relu" else init_leaky)
        if args.max:
            self.post_model= nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        elif args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Flatten()
            flat_features = 16*4*args.channels

        if not self.args.linear_repr:
            self.last_fc = nn.Identity()
        elif args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # nn.Sigmoid(),
                nn.LeakyReLU(),
                drop_func() if args.ext_dropout else nn.Identity,
                # drop_func(),
                nn.Linear(args.neurons_rep, num_output)
            )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        if args.projection:
            # self.projector = MLPHead(num_output, args.neurons_rep, num_output)
            self.projector = MLP(num_output if self.args.linear_repr else flat_features, num_output, args.neurons_rep, args.proj_activation, args.proj_layers, last_linear=True,dropout=args.proj_dropout, batch_norm=args.batchnorm)

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


class ConvNet512_3(nn.Module):
    def __init__(self, args, num_output, dropout=True):
        super().__init__()
        self.args=args
        if args.activation == "relu":
            self.activ_func = nn.ReLU
        if args.activation == "leaky":
            self.activ_func = nn.LeakyReLU
        if args.activation == "elu":
            self.activ_func = nn.ELU

        if args.dropout == 0 or args.dropout == 3 or not dropout:
            drop_func = nn.Identity
        elif args.dropout == 1:
            drop_func = lambda : nn.Dropout(p=args.drop_val)
        elif args.dropout == 2:
            drop_func = lambda : nn.Dropout2d(p=args.drop_val)

        if not args.batchnorm:
            batchnorm_func = nn.Identity
            batchnorm_func2 = nn.Identity
        else:
            batchnorm_func = nn.BatchNorm2d
            batchnorm_func2 = nn.BatchNorm1d

        channels = 3 if not self.args.binocular else 6
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=args.channels, kernel_size=8, stride=4, padding=2),  # 128 - 8 / 4 = 120 / 4 = 30 + 1
            batchnorm_func(args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=args.channels, out_channels=2*args.channels, kernel_size=4, stride=2, padding=1),  # 31 - 4 / 2 = 13 + 1
            batchnorm_func(2 * args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=2*args.channels, out_channels=4*args.channels, kernel_size=4, stride=2, padding=1),  # 14 - 2 / 2 =  6 + 1
            batchnorm_func(4 * args.channels),
            self.activ_func(),
            drop_func(),
            nn.Conv2d(in_channels=4*args.channels, out_channels=8*args.channels, kernel_size=4, stride=2, padding=1),  # 6 - 2 / 2 = 2 + 1
            batchnorm_func(8 * args.channels),
            self.activ_func(),
            drop_func() if args.dropout == 1 else nn.Identity()
        )

        self.model.apply(init_relu if args.activation == "relu" else init_leaky)
        if args.max:
            self.post_model= nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        elif args.average:
            self.post_model= nn.Sequential(nn.AvgPool2d(4), nn.Flatten())
            # self.post_model= nn.Sequential(nn.AdaptiveAvgPool2d(coef_channels*128), nn.Flatten())
            flat_features = 4*args.channels
        else:
            self.post_model= nn.Flatten()
            flat_features = 16*4*args.channels

        if not self.args.linear_repr:
            self.last_fc = nn.Identity()
        elif args.extended:
            self.last_fc = nn.Sequential(
                nn.Linear(flat_features, args.neurons_rep),
                # nn.ReLU(inplace=True),
                # nn.Sigmoid(),
                batchnorm_func2(args.neurons_rep),
                nn.ReLU(),
                drop_func() if args.ext_dropout else nn.Identity,
                nn.Linear(args.neurons_rep, num_output)
            )
        else:
            if not args.tanh:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output))
            else:
                self.last_fc = nn.Sequential(nn.Linear(flat_features, num_output), nn.Tanh())

        self.repr = num_output if self.args.linear_repr else flat_features
        if args.projection:
            # self.projector = MLPHead(num_output, args.neurons_rep, num_output)
            self.projector = MLP(self.repr, num_output, args.neurons_rep, args.proj_activation, args.proj_layers, last_linear=True,dropout=args.proj_dropout, batch_norm=args.batchnorm)

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
    def __init__(self, args, in_channels, mlp_hidden_size, projection_size, batchnorm=False, layernorm=False, dropout=False):
        super(MLPHead, self).__init__()
        if args.proj_layers ==1:
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size) if batchnorm else nn.Identity(),
                nn.LayerNorm(mlp_hidden_size) if layernorm else nn.Identity(),
                nn.Dropout(0.5) if dropout else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )
        elif args.proj_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(in_channels, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size) if batchnorm else nn.Identity(),
                nn.LayerNorm(mlp_hidden_size) if layernorm else nn.Identity(),
                nn.Dropout(0.5) if dropout else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, mlp_hidden_size),
                nn.BatchNorm1d(mlp_hidden_size) if batchnorm else nn.Identity(),
                nn.LayerNorm(mlp_hidden_size) if layernorm else nn.Identity(),
                nn.Dropout(0.5) if dropout else nn.Identity(),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, projection_size)
            )

    def forward(self, x):
        return self.net(x)
