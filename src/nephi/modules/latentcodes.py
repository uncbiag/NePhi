import torch
import torch.nn as nn
import math
from nephi.gridsample_diff.cuda_gridsample import grid_sample_2d, grid_sample_3d

class HybridLatentCode(nn.Module):
    def __init__(self, global_latent, local_latent) -> None:
        """
        global_latent: B x feat_dim
        local_latent: B x feat_dim x H x W
        """
        super().__init__()
        self.global_latent = global_latent
        self.local_latent = local_latent
        self.device = self.global_latent.device
        self.batch_size = global_latent.shape[0]

        self.grid_sampler = (
            grid_sample_2d if len(local_latent.shape[2:]) == 2 else grid_sample_3d
        )

    def forward(self, x=None):
        if x is None:
            N = torch.prod(torch.tensor(self.local_latent.shape[2:]))
            return torch.cat(
                [
                    self.global_latent.unsqueeze(-1).expand(-1, -1, N),
                    self.local_latent.view(*self.local_latent.shape[:2], -1),
                ],
                dim=1,
            ).permute(0, 2, 1)

        assert (
            x.shape[0] == 1 or x.shape[0] == self.global_latent.shape[0]
        ), "Batch size does not match."

        # Match batch size
        if x.shape[0] != self.global_latent.shape[0]:
            x = x.expand(self.global_latent.shape[0], -1, -1)

        B, N, dim = x.shape
        for _ in range(dim - 1):
            x = x.unsqueeze(1)
        return torch.cat(
            [
                self.global_latent.unsqueeze(-1).expand(-1, -1, N),
                self.grid_sampler(
                    self.local_latent,
                    x.flip(-1),
                    padding_mode="border",
                    align_corners=True,
                ).view(B, -1, N),
            ],
            dim=1,
        ).permute(0, 2, 1)

    def reg(self):
        return torch.mean(torch.norm(self.global_latent, dim=1)) + torch.mean(
            torch.norm(self.local_latent, dim=1)
        )

class LatentCodeTable(nn.Module):
    def __init__(
        self,
        global_dim,
        local_dim,
        data_num,
        local_shape,
        global_max=1.0,
        local_max=1.0,
    ) -> None:
        super().__init__()
        self.register_parameter(
            "global_latent",
            nn.Parameter(
                torch.normal(0.0, 1.0 / math.sqrt(global_dim), (data_num, global_dim))
            ),
        )
        self.register_parameter(
            "local_latent",
            nn.Parameter(
                torch.normal(
                    0.0, 1.0 / math.sqrt(local_dim), (data_num, local_dim, *local_shape)
                )
            ),
        )

    def forward(self, idx):
        return HybridLatentCode(self.global_latent[idx], self.local_latent[idx])

    def reg(self, idx):
        return torch.mean(torch.norm(self.global_latent[idx], dim=1)) + torch.mean(
            torch.norm(self.local_latent[idx], dim=1)
        )

