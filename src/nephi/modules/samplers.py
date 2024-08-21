import torch.nn as nn
from nephi.utils import sampler

class RandomSampler(nn.Module):
    def __init__(self, sample_num, dim=2, with_frame=True) -> None:
        super().__init__()
        self.sample_num = sample_num
        self.dim = dim
        self.with_frame = with_frame

    def forward(self, batch_size=1, device="cpu"):
        """
        Return 1 x N x space_dim
        """
        return (
            sampler(num=self.sample_num, dim=self.dim, within_frame=self.with_frame)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(device)
        )