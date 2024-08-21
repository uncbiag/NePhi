import torch
import torch.nn as nn
from nephi.modules.layers import convBlock, FullyConnectBlock
from nephi.modules.latentcodes import HybridLatentCode
from nephi.utils import compute_encoder_output_shape


class GlobalAndLocalEncoder(nn.Module):
    def __init__(
        self,
        latent_dim,
        data_shape,
        enc_filters=[16, 32, 64, 128, 256],
        local_feature_layer_num=3,
        kernel_size=3,
        batchnorm=True,
    ) -> None:
        super().__init__()

        dim = len(data_shape)
        self.local_encoder = convBlock(
            enc_filters[local_feature_layer_num],
            latent_dim,
            dim=dim,
            batchnorm=batchnorm,
        )

        feature_net = nn.ModuleList()
        enc_filters = [2] + enc_filters
        for i in range(len(enc_filters) - 1):
            feature_net.append(
                convBlock(
                    enc_filters[i],
                    enc_filters[i + 1],
                    kernel_size=kernel_size,
                    stride=2,
                    padding=kernel_size // 2,
                    bias=True,
                    dim=dim,
                    batchnorm=batchnorm,
                )
            )
        self.feature_encoder = feature_net

        global_net = nn.ModuleList()
        global_net.append(nn.Flatten())
        dim = compute_encoder_output_shape([2] * (len(enc_filters) - 1), data_shape)
        print(f"Dim before FC: {dim}")
        global_net.append(FullyConnectBlock(enc_filters[-1] * dim, 256))
        global_net.append(FullyConnectBlock(256, latent_dim, nonlinear=None))
        self.global_encoder = nn.Sequential(*global_net)
        self.local_feature_layer_num = (
            local_feature_layer_num
            if local_feature_layer_num >= 0
            else len(enc_filters) - 1 + local_feature_layer_num
        )
        self.latent_dim = latent_dim * 2

    def forward(self, I1, I2):
        """
        x: B x N x space_dim
        Return BxNxC.
        """
        x = torch.cat([I1, I2], dim=1)
        local_features = None
        for i in range(len(self.feature_encoder)):
            x = self.feature_encoder[i](x)
            if i == self.local_feature_layer_num:
                local_features = x
        latent_code_global = self.global_encoder(x)  # BxC
        latent_code_local = self.local_encoder(local_features)  # BxCxHxW

        return HybridLatentCode(latent_code_global, latent_code_local)