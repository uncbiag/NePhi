import torch
import numpy as np
import torch.nn as nn
import math

# from torchmeta.modules import (MetaModule, MetaSequential)
# from torchmeta.modules.utils import get_subdict
from collections import OrderedDict

# from modsiren.resnet_encoder import *
# from pytorch_lightning import Trainer, seed_everything

# seed_everything(42)


# This script is from https://github.com/ishit/modsiren

class PositionEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=10):
        super(PositionEmbedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        # TODO(mgharbi): Pi missing here?
        self.freq_bands = torch.cat([2 ** torch.linspace(0, N_freqs - 1, N_freqs)])

    def forward(self, x):
        out = [x]

        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, N_freqs=256, scale=10.0):
        super(FourierEmbedding, self).__init__()
        self.B = scale * torch.randn((in_channels, N_freqs))
        self.B = torch.nn.Parameter(self.B, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x.matmul(self.B.to(x.device))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PosEncodingNeRF(nn.Module):
    """Module to add positional encoding as in NeRF [Mildenhall et al. 2020]."""

    def __init__(self, in_features, sidelength=None, fn_samples=None, use_nyquist=True):
        super().__init__()

        self.in_features = in_features

        if self.in_features == 3:
            self.num_frequencies = 10
        elif self.in_features == 2:
            assert sidelength is not None
            if isinstance(sidelength, int):
                sidelength = (sidelength, sidelength)
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(
                    min(sidelength[0], sidelength[1])
                )
        elif self.in_features == 1:
            assert fn_samples is not None
            self.num_frequencies = 4
            if use_nyquist:
                self.num_frequencies = self.get_num_frequencies_nyquist(fn_samples)

        self.out_dim = in_features + 2 * in_features * self.num_frequencies

    def get_num_frequencies_nyquist(self, samples):
        nyquist_rate = 1 / (2 * (2 * 1 / samples))
        return int(math.floor(math.log(nyquist_rate, 2)))

    def forward(self, coords):
        coords = coords.view(coords.shape[0], -1, self.in_features)

        coords_pos_enc = coords
        for i in range(self.num_frequencies):
            for j in range(self.in_features):
                c = coords[..., j]

                sin = torch.unsqueeze(torch.sin((2**i) * np.pi * c), -1)
                cos = torch.unsqueeze(torch.cos((2**i) * np.pi * c), -1)

                coords_pos_enc = torch.cat((coords_pos_enc, sin, cos), axis=-1)

        return coords_pos_enc.reshape(coords.shape[0], -1, self.out_dim)


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, "weight"):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


# def init_weights_normal(m):
#     if type(m) == BatchLinear or type(m) == nn.Linear:
#         if hasattr(m, 'weight'):
#             nn.init.kaiming_normal_(
#                 m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        m.weight.data = m.weight.data / 1.0e2

    if hasattr(m, "bias"):
        with torch.no_grad():
            m.bias.uniform_(-1 / in_features_main_net, 1 / in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, "weight"):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity="relu", mode="fan_in")
        m.weight.data = m.weight.data / 1.0e2

    if hasattr(m, "bias"):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1 / fan_in, 1 / fan_in)


# class BatchLinear(nn.Linear, MetaModule):
#     __doc__ = nn.Linear.__doc__

#     def forward(self, input, params=None):
#         if params is None:
#             params = OrderedDict(self.named_parameters())

#         bias = params.get('bias', None)
#         weight = params['weight']

#         output = input.matmul(weight.permute(
#             *[i for i in range(len(weight.shape) - 2)], -1, -2))
#         output += bias.unsqueeze(-2)
#         return output


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
        residual=False,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.residual = residual

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        if self.residual:
            return torch.sin(self.omega_0 * self.linear(input)) + input
        else:
            return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        PE_side,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
        latent_dim=0,
    ):
        super().__init__()

        self.pos_encoder = PosEncodingNeRF(in_features, PE_side)
        print(f"PE out dim: {self.pos_encoder.out_dim}")

        self.net = []
        if latent_dim > 0:
            self.use_latent_code = True
            self.net.append(
                SineLayer(
                    self.pos_encoder.out_dim + latent_dim,
                    hidden_features,
                    is_first=True,
                    omega_0=first_omega_0,
                )
            )
        else:
            self.net.append(
                SineLayer(
                    self.pos_encoder.out_dim,
                    hidden_features,
                    is_first=True,
                    omega_0=first_omega_0,
                )
            )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                # final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                #                               np.sqrt(6 / hidden_features) / hidden_omega_0)
                final_linear.weight.uniform_(0, 1e-5)
                final_linear.bias.uniform_(0.0, 0.0)

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, latent_code=None):
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input

        if self.use_latent_code:
            if len(latent_code.shape) != len(coords.shape):
                BS, PS, D = coords.shape
            latent_code = latent_code.unsqueeze(-2).repeat(1, PS, 1)
            output = self.net(
                torch.cat([self.pos_encoder(coords), latent_code], dim=-1)
            )
        else:
            output = self.net(self.pos_encoder(coords))
        return {"model_in": coords, "model_out": output, "latent_vec": latent_code}


class SineMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers,
        hidden_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords, gating_layers=None):
        output = self.net(coords)
        return output


class GatedSineMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers,
        hidden_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
        residual=False,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.first_layer = SineLayer(
            in_features,
            hidden_features,
            is_first=True,
            omega_0=first_omega_0,
            residual=False,
        )

        for i in range(hidden_layers):
            layer = SineLayer(
                hidden_features,
                hidden_features,
                is_first=False,
                omega_0=hidden_omega_0,
                residual=residual,
            )
            setattr(self, f"hidden_layer_{i}", layer)

        if outermost_linear is None:
            self.final_layer = None
        else:
            if outermost_linear:
                final_linear = nn.Linear(hidden_features, out_features)

                with torch.no_grad():
                    final_linear.weight.uniform_(
                        -np.sqrt(6 / hidden_features) / hidden_omega_0,
                        np.sqrt(6 / hidden_features) / hidden_omega_0,
                    )

                self.final_layer = final_linear
            else:
                self.final_layer = SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    residual=residual,
                )

    def forward(self, coords, gating_layers):
        out = self.first_layer(coords)
        for i in range(self.hidden_layers):
            out = gating_layers[i] * getattr(self, f"hidden_layer_{i}")(out)
        if self.final_layer is None:
            output = out
        else:
            output = self.final_layer(out)
        return output


class GatedReluMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers,
        hidden_features,
        outermost_linear=False,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.ReLU(True)
        )

        for i in range(hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nn.ReLU(True)
            )
            setattr(self, f"hidden_layer_{i}", layer)

        self.final_layer = nn.Sequential(nn.Linear(hidden_features, out_features))

    def forward(self, coords, gating_layers):
        out = self.first_layer(coords)
        for i in range(self.hidden_layers):
            out = gating_layers[i] * getattr(self, f"hidden_layer_{i}")(out)
        output = self.final_layer(out)
        return output


class ReLUMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, hidden_features):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features), nn.ReLU()
        )
        self.hidden_layers = hidden_layers

        for i in range(hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_features, hidden_features), nn.ReLU()
            )
            setattr(self, f"hidden_layer_{i}", layer)

        self.final_layer = nn.Sequential(nn.Linear(hidden_features, out_features))

    def forward(self, x, gating_layers=None):
        out = self.first_layer(x)
        for i in range(self.hidden_layers):
            if gating_layers:
                out = gating_layers[i] * getattr(self, f"hidden_layer_{i}")(out)
            else:
                out = getattr(self, f"hidden_layer_{i}")(out)

        output = self.final_layer(out)
        return output


class ModulationMLP(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, bias=True):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Linear(in_features, hidden_features, bias=bias), nn.ReLU(True)
        )
        self.hidden_layers = hidden_layers  # since there is no final layer
        for i in range(self.hidden_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_features, hidden_features, bias=bias), nn.ReLU(True)
            )
            setattr(self, f"layer_{i}", layer)

    def forward(self, coords):
        output = self.first_layer(coords)
        skip = output
        gating_layers = []
        for i in range(self.hidden_layers):
            output = getattr(self, f"layer_{i}")(output) + skip
            gating_layers.append(output)
        return gating_layers


class SineModulationMLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.hidden_layers = hidden_layers
        self.first_layer = SineLayer(
            in_features, hidden_features, is_first=True, omega_0=first_omega_0
        )

        for i in range(hidden_layers):
            layer = SineLayer(
                hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0
            )
            setattr(self, f"layer_{i}", layer)

    def forward(self, coords):
        output = self.first_layer(coords)
        skip = output
        gating_layers = []
        for i in range(self.hidden_layers):
            output = getattr(self, f"layer_{i}")(output) + skip
            gating_layers.append(output)
        return gating_layers


class Conv2dResBlock(nn.Module):
    """Aadapted from https://github.com/makora9143/pytorch-convcnp/blob/master/convcnp/modules/resblock.py"""

    def __init__(self, in_channel, out_channel=128):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2),
            nn.ReLU(),
        )

        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output


class ConvImgEncoder(nn.Module):
    # Try vectorized patch
    # Add stride or pooling
    # Limit the number of parameters
    # learn AE and see the limit of the encoder

    def __init__(self, channel, image_resolution, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 128, 3, 2, 1),  # 4 x 4
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(
            128 * ((image_resolution // 8) ** 2), latent_dim
        )  # Ton of parameters
        # self.fc = nn.Linear(128 * (4 ** 2), latent_dim) # Ton of parameters
        self.image_resolution = image_resolution

    def forward(self, model_input):
        o = self.relu(self.conv_theta(model_input))
        o = self.cnn(o)
        o = self.fc(self.relu_2(o).view(o.shape[0], -1))
        return o


class HyperConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution):
        super().__init__()

        # conv_theta is input convolution
        self.conv_theta = nn.Conv2d(channel, 128, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)

        self.cnn = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            Conv2dResBlock(256, 256),
            nn.Conv2d(256, 256, 1, 1, 0),
        )

        self.relu_2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(image_resolution**2, 1)
        self.image_resolution = image_resolution

    def forward(self, I):
        o = self.relu(self.conv_theta(I))
        o = self.cnn(o)
        print(o.shape)
        exit()

        o = self.fc(self.relu_2(o).view(o.shape[0], 256, -1)).squeeze(-1)
        return o


class SimpleConvImgEncoder(nn.Module):
    def __init__(self, channel, image_resolution, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(channel, 128, 3, 2, 1),  # 16 x 16
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),  # 8 x 8
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 4 x 4
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 2 x 2
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),  # 1 x 1
            nn.ReLU(),
        )
        self.fc = nn.Linear(256, latent_dim)

    def forward(self, model_input):
        o = self.cnn(model_input)
        o = self.fc(o.view(o.shape[0], -1))
        return o


class Encoder3(nn.Module):
    def __init__(self, in_channels, image_resolution, latent_dim, batch_norm=False):
        super().__init__()
        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [2 ** (i + 5) for i in range(int(np.log2(image_resolution)) - 1)]

        # Build Encoder
        for h_dim in hidden_dims:
            if batch_norm:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels=h_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.BatchNorm2d(h_dim),
                        nn.LeakyReLU(),
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            out_channels=h_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1,
                        ),
                        nn.LeakyReLU(),
                    )
                )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc = nn.Linear(hidden_dims[-1] * 4, latent_dim)

    def forward(self, x):
        res = self.encoder(x)
        res = torch.flatten(res, start_dim=1)

        return self.fc(res)


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    # taken from occupancy networks

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)
        c = self.fc_c(self.actvn(net))

        return c


class Decoder3(nn.Module):
    def __init__(self, in_channels, image_resolution, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        modules = []
        hidden_dims = [2 ** (i + 5) for i in range(int(np.log2(image_resolution)) - 1)]
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)
        hidden_dims.reverse()

        self.hidden_dims = hidden_dims
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        res = self.decoder_input(z)
        res = res.view(-1, self.hidden_dims[0], 2, 2)
        res = self.decoder(res)
        res = self.final_layer(res)
        return res


class AE(nn.Module):
    def __init__(self, out_features, latent_dim, patch_res):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder3(out_features, patch_res[0], latent_dim)
        self.decoder = Decoder3(out_features, patch_res[0], latent_dim)

    def forward(self, model_input):
        in_img = model_input["img"].cuda().float()
        latent_vec = self.encoder(in_img)
        out = self.decoder(latent_vec)
        model_output = out.permute(0, 2, 3, 1).reshape(
            in_img.shape[0], -1, in_img.shape[1]
        )
        return {"model_in": 0, "model_out": model_output, "latent_vec": latent_vec}


# class VideoEncoder(nn.Module):
# def __init__(self, in_channels, patch_res, latent_dim):
# super(VideoEncoder, self).__init__()
# self.latent_dim = latent_dim
# modules = []
# # hidden_dims = [2 ** (i + 5)
# # for i in range(int(np.log2(patch_res[1])) - 1)]
# hidden_dims = [512
# for i in range(int(np.log2(patch_res[1])) - 1)]

# # Build Encoder
# for h_dim in hidden_dims:
# modules.append(
# nn.Sequential(
# nn.Conv3d(in_channels, out_channels=h_dim,
# kernel_size=3, stride=2, padding=1),
# nn.LeakyReLU())
# )
# in_channels = h_dim

# self.encoder = nn.Sequential(*modules)
# self.fc = nn.Linear(hidden_dims[-1]*4, latent_dim)

# def forward(self, x):
# res = self.encoder(x)
# res = torch.flatten(res, start_dim=1)
# return self.fc(res)


class VideoEncoder(nn.Module):
    def __init__(self, in_channels, patch_res, latent_dim):
        super(VideoEncoder, self).__init__()

        im_latent_dim = 256
        self.encoder = Encoder3(in_channels, patch_res[1], im_latent_dim)
        self.max_pool = nn.MaxPool1d(patch_res[0])

        self.fc1 = nn.Linear(im_latent_dim * patch_res[0], latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

        self.fc = nn.Sequential(self.fc1, nn.LeakyReLU(), self.fc2)

    def forward(self, x):
        x_2d = x.permute(0, 2, 1, 3, 4)
        BS, F, C, H, W = x_2d.shape
        input_2d = x_2d.reshape(BS * F, C, H, W)
        output_2d = self.encoder(input_2d)  # BS * F, latent_dim
        input_3d = output_2d.view(BS, F, -1)
        input_3d = input_3d.reshape(BS, -1)
        # input_3d = input_3d.permute(0, 2, 1)
        # pooled_3d = self.max_pool(input_3d).squeeze(-1)
        # output_3d = self.fc(pooled_3d)
        output_3d = self.fc(input_3d)

        return output_3d


class LocalMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers=4,
        hidden_features=256,
        latent_dim=None,
        synthesis_activation=None,
        modulation_activation=None,
        concat=True,
        embedding=None,
        freq_scale=1.0,
        N_freqs=None,
        encoder=False,
        encoder_type="simple",
        patch_res=32,
        residual=False,
    ):
        super().__init__()

        if embedding == "ffn":
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(in_features, N_freqs, scale=freq_scale)
        elif embedding == "pe":
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        elif isinstance(embedding, PosEncodingNeRF):
            self.embed = embedding
            total_in_features = self.embed.out_dim
        else:
            self.embed = None
            total_in_features = in_features

        self.concat = concat

        if modulation_activation:
            if synthesis_activation == "sine":
                first_omega_0 = 30.0 / freq_scale
                hidden_omega_0 = 30.0 / freq_scale
                self.synthesis_nw = GatedSineMLP(
                    total_in_features,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                    first_omega_0=freq_scale,
                    hidden_omega_0=hidden_omega_0,
                    residual=residual,
                )
            elif synthesis_activation == "relu":
                self.synthesis_nw = GatedReluMLP(
                    total_in_features,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                )
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features, out_features, hidden_layers, hidden_features
                )

            if modulation_activation == "relu":
                if concat:
                    self.modulation_nw = ModulationMLP(
                        latent_dim + total_in_features, hidden_features, hidden_layers
                    )
                else:
                    self.modulation_nw = ModulationMLP(
                        latent_dim, hidden_features, hidden_layers, bias=False
                    )

            elif modulation_activation == "sine":
                if concat:
                    self.modulation_nw = SineModulationMLP(
                        latent_dim + total_in_features, hidden_features, hidden_layers
                    )
                else:
                    self.modulation_nw = SineModulationMLP(
                        latent_dim, hidden_features, hidden_layers
                    )
            else:
                print("Modulation sine not implemented yet!")
                exit()
        else:
            self.modulation_nw = None
            if synthesis_activation == "sine":
                self.synthesis_nw = SineMLP(
                    total_in_features + latent_dim,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                )
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features + latent_dim,
                    out_features,
                    hidden_layers,
                    hidden_features,
                )

        if encoder:
            if len(patch_res) > 2:
                self.encoder = VideoEncoder(out_features, patch_res, latent_dim)
            elif len(patch_res) == 0:
                self.encoder = SimplePointnet(c_dim=latent_dim)
            else:
                if encoder_type == "simple":
                    self.encoder = Encoder3(out_features, patch_res[0], latent_dim)
                else:
                    self.encoder = ConvImgEncoder(
                        out_features, patch_res[0], latent_dim
                    )
        else:
            self.encoder = None

    def forward(self, coords, embedding):  # def forward(self, model_input):
        # coords = model_input['coords'].float()

        # if self.encoder:
        #     latent_vec = self.encoder(model_input['img'].cuda().float())
        # else:
        #     latent_vec = model_input['embedding'].float()

        coords = coords.float()
        latent_vec = embedding.float()

        if len(latent_vec.shape) != len(coords.shape):
            BS, PS, D = coords.shape
            latent_vec = latent_vec.unsqueeze(-2).repeat(1, PS, 1)

        if self.embed:
            coords = self.embed(coords)

        gating_layers = None
        if self.modulation_nw:
            if self.concat:
                gating_layers = self.modulation_nw(
                    torch.cat([latent_vec, coords], dim=-1)
                )
            else:
                gating_layers = self.modulation_nw(latent_vec)
            model_output = self.synthesis_nw(coords, gating_layers)

        else:
            model_output = self.synthesis_nw(
                torch.cat([latent_vec, coords], dim=-1), gating_layers
            )

        return {"model_in": coords, "model_out": model_output, "latent_vec": latent_vec}


class PatchMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers=4,
        hidden_features=256,
        latent_dim=None,
        synthesis_activation=None,
        modulation_activation=None,
        concat=True,
        embedding=None,
        freq_scale=1.0,
        N_freqs=None,
        encoder=False,
        encoder_type="simple",
        patch_res=32,
        residual=False,
    ):
        super().__init__()

        if embedding == "ffn":
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(in_features, N_freqs, scale=freq_scale)
        elif embedding == "pe":
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        elif isinstance(embedding, PosEncodingNeRF):
            self.embed = embedding
            total_in_features = self.embed.out_dim
        else:
            self.embed = None
            total_in_features = in_features

        print(total_in_features)
        self.concat = concat

        if modulation_activation:
            if synthesis_activation == "sine":
                first_omega_0 = 30.0 / freq_scale
                hidden_omega_0 = 30.0 / freq_scale
                self.synthesis_nw = GatedSineMLP(
                    total_in_features,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                    first_omega_0=freq_scale,
                    hidden_omega_0=hidden_omega_0,
                    residual=residual,
                )
            elif synthesis_activation == "relu":
                self.synthesis_nw = GatedReluMLP(
                    total_in_features,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                )
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features, out_features, hidden_layers, hidden_features
                )

            if modulation_activation == "relu":
                if concat:
                    self.modulation_nw = ModulationMLP(
                        latent_dim + total_in_features, hidden_features, hidden_layers
                    )
                else:
                    self.modulation_nw = ModulationMLP(
                        latent_dim, hidden_features, hidden_layers, bias=False
                    )

            elif modulation_activation == "sine":
                if concat:
                    self.modulation_nw = SineModulationMLP(
                        latent_dim + total_in_features, hidden_features, hidden_layers
                    )
                else:
                    self.modulation_nw = SineModulationMLP(
                        latent_dim, hidden_features, hidden_layers
                    )
            else:
                print("Modulation sine not implemented yet!")
                exit()
        else:
            self.modulation_nw = None
            if synthesis_activation == "sine":
                self.synthesis_nw = SineMLP(
                    total_in_features + latent_dim,
                    out_features,
                    hidden_layers,
                    hidden_features,
                    outermost_linear=True,
                )
            else:
                self.synthesis_nw = ReLUMLP(
                    total_in_features + latent_dim,
                    out_features,
                    hidden_layers,
                    hidden_features,
                )

        if encoder:
            if len(patch_res) > 2:
                self.encoder = VideoEncoder(out_features, patch_res, latent_dim)
            elif len(patch_res) == 0:
                self.encoder = SimplePointnet(c_dim=latent_dim)
            else:
                if encoder_type == "simple":
                    self.encoder = Encoder3(out_features, patch_res[0], latent_dim)
                else:
                    self.encoder = ConvImgEncoder(
                        out_features, patch_res[0], latent_dim
                    )
        else:
            self.encoder = None

    def forward(self, coords, embedding):  # def forward(self, model_input):
        # coords = model_input['coords'].float()

        # if self.encoder:
        #     latent_vec = self.encoder(model_input['img'].cuda().float())
        # else:
        #     latent_vec = model_input['embedding'].float()

        coords = coords.float()
        latent_vec = embedding.float()

        if len(latent_vec.shape) != len(coords.shape):
            BS, PS, D = coords.shape
            latent_vec = latent_vec.unsqueeze(-2).repeat(1, PS, 1)

        if self.embed:
            coords = self.embed(coords)

        gating_layers = None
        if self.modulation_nw:
            if self.concat:
                gating_layers = self.modulation_nw(
                    torch.cat([latent_vec, coords], dim=-1)
                )
            else:
                gating_layers = self.modulation_nw(latent_vec)
            model_output = self.synthesis_nw(coords, gating_layers)

        else:
            model_output = self.synthesis_nw(
                torch.cat([latent_vec, coords], dim=-1), gating_layers
            )

        return {"model_in": coords, "model_out": model_output, "latent_vec": latent_vec}


class GlobalMLP(nn.Module):
    def __init__(
        self,
        in_features=2,
        out_features=1,
        hidden_layers=4,
        hidden_features=256,
        synthesis_activation=None,
        embedding=None,
        N_freqs=None,
        freq_scale=1.0,
    ):
        super().__init__()

        if embedding == "ffn":
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(in_features, N_freqs, scale=freq_scale)
        elif embedding == "pe":
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        else:
            self.embed = None
            total_in_features = in_features

        if synthesis_activation == "sine":
            first_omega_0 = 30.0 / freq_scale
            hidden_omega_0 = 30.0 / freq_scale
            print(first_omega_0)
            self.synthesis_nw = SineMLP(
                total_in_features,
                out_features,
                hidden_layers,
                hidden_features,
                outermost_linear=True,
                first_omega_0=first_omega_0,
                hidden_omega_0=hidden_omega_0,
            )
        else:
            self.synthesis_nw = ReLUMLP(
                total_in_features, out_features, hidden_layers, hidden_features
            )

    def forward(self, model_input):
        coords = model_input["global_coords"]
        if self.embed:
            coords = self.embed(coords)
        model_output = self.synthesis_nw(coords, None)
        return {"model_in": coords, "model_out": model_output, "latent_vec": 0.0}


class MultiResMLP(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_layers=4,
        hidden_features=256,
        latent_dim_global=None,
        latent_dim_local=None,
        synthesis_activation=None,
        modulation_activation=None,
        embedding=None,
        freq_scale=1.0,
        N_freqs=None,
        encoder=False,
        encoder_type="simple",
        patch_res=32,
        residual=False,
    ):
        super().__init__()

        if embedding == "ffn":
            total_in_features = N_freqs * 2
            self.embed = FourierEmbedding(in_features, N_freqs, scale=freq_scale)
        elif embedding == "pe":
            total_in_features = N_freqs * 2 * in_features + in_features
            self.embed = PositionEmbedding(in_features, N_freqs)
        elif isinstance(embedding, PosEncodingNeRF):
            self.embed = embedding
            total_in_features = self.embed.out_dim
        else:
            self.embed = None
            total_in_features = in_features

        print(total_in_features)

        if synthesis_activation == "sine":
            first_omega_0 = 30.0 / freq_scale
            hidden_omega_0 = 30.0 / freq_scale
            self.synthesis_nw_local = GatedSineMLP(
                total_in_features,
                hidden_features,
                hidden_layers,
                hidden_features,
                outermost_linear=None,
                first_omega_0=freq_scale,
                hidden_omega_0=hidden_omega_0,
                residual=residual,
            )
        elif synthesis_activation == "relu":
            self.synthesis_nw_local = GatedReluMLP(
                total_in_features,
                hidden_features,
                hidden_layers,
                hidden_features,
                outermost_linear=None,
            )
        else:
            self.synthesis_nw_local = ReLUMLP(
                total_in_features, hidden_features, hidden_layers, hidden_features
            )

        self.synthesis_nw_global = GatedSineMLP(
            hidden_features,
            out_features,
            hidden_layers,
            hidden_features,
            outermost_linear=True,
            first_omega_0=freq_scale,
            hidden_omega_0=hidden_omega_0,
            residual=residual,
        )

        if modulation_activation == "relu":
            self.modulation_nw_local = ModulationMLP(
                latent_dim_local + total_in_features,
                hidden_features,
                hidden_layers,
                bias=False,
            )

            self.modulation_nw_global = ModulationMLP(
                latent_dim_global + hidden_features, hidden_features, hidden_layers
            )

        elif modulation_activation == "sine":
            self.modulation_nw_local = SineModulationMLP(
                latent_dim_local, hidden_features, hidden_layers
            )
            self.modulation_nw_global = SineModulationMLP(
                latent_dim_global, hidden_features, hidden_layers
            )
        else:
            print("Modulation sine not implemented yet!")
            exit()

    def forward(self, coords, embedding):  # def forward(self, model_input):
        coords = coords.float()
        latent_vec_g, latent_vec_l = embedding
        latent_vec_g, latent_vec_l = latent_vec_g.float(), latent_vec_l.float()

        if len(latent_vec_g.shape) != len(coords.shape):
            BS, PS, D = coords.shape
            latent_vec_g = latent_vec_g.unsqueeze(-2).repeat(1, PS, 1)
            latent_vec_l = latent_vec_l.unsqueeze(-2).repeat(1, PS, 1)

        if self.embed:
            coords = self.embed(coords)

        model_output = self.synthesis_nw_local(
            coords, self.modulation_nw_local(torch.cat([latent_vec_l, coords], dim=-1))
        )
        model_output = self.synthesis_nw_global(
            model_output,
            self.modulation_nw_global(torch.cat([latent_vec_g, model_output], dim=-1)),
        )

        return {"model_in": coords, "model_out": model_output}

if __name__ == "__main__":
    # Test encoder
    # encoder = ConvImgEncoder(3, 32, 128)
    # encoder = VideoEncoder([10, 32, 32], 128)
    encoder = VideoEncoder(in_channels=3, patch_res=[10, 32, 32], latent_dim=128)
    # test = torch.rand((64, 3, 32, 32))
    test = torch.rand((64, 3, 10, 32, 32))
    o = encoder(test)
    breakpoint()

    e = SimplePointnet(c_dim=256)
    # def __init__(self, c_dim=128, dim=3, hidden_dim=128):
    test = torch.rand((64, 1024, 3))
    ret = e.forward(test)
    print(ret.shape)
