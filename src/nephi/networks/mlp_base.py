import torch
import torch.nn as nn

from nephi.utils import warp_with_latent_code


class OptBasedRegistrationMLPBase(nn.Module):
    def __init__(
        self, phi, latent_codes, sim_sampler, reg_sampler, sim, lamb, prior_lamb
    ) -> None:
        super().__init__()
        self.phi = phi
        self.lamb = lamb
        self.sim_sampler = sim_sampler
        self.reg_sampler = reg_sampler
        self.sim = sim
        self.latent_codes = latent_codes
        self.prior_lamb = prior_lamb

        with torch.no_grad():
            # Init the displacement from zero
            torch.nn.init.zeros_(self.phi.synthesis_nw.final_layer.weight)
            torch.nn.init.zeros_(self.phi.synthesis_nw.final_layer.bias)

    def forward(self, I1, I2, idx):
        latent_code = self.latent_codes(idx)
        sim_loss = self.compute_sim(I1, I2, latent_code)
        reg_loss = self.reg(latent_code)
        prior_loss = self.prior_reg(idx)
        loss = sim_loss + self.lamb * reg_loss + prior_loss * self.prior_lamb
        return loss, {"sim_loss": sim_loss.detach(), "reg_loss": reg_loss.detach()}

    def reg(self, latent_code):
        return torch.zeros((1, 1), device=latent_code.device)

    def prior_reg(self, idx):
        return self.latent_codes.reg(idx)

    def compute_sim(self, I1, I2, latent_code):
        x = self.sim_sampler(
            batch_size=latent_code.batch_size, device=latent_code.device
        )
        phi = self.phi(x, latent_code(x))["model_out"] + x
        for _ in range(len(I1.shape[2:]) - 1):
            phi = phi.unsqueeze(1)
            x = x.unsqueeze(1)
        warped_img = torch.nn.functional.grid_sample(
            I1, phi.flip(-1), align_corners=True
        )
        target = torch.nn.functional.grid_sample(I2, x.flip(-1), align_corners=True)
        return self.sim(warped_img, target)

    def register(self, I1, I2, coords, idx, I1_seg=None):
        with torch.no_grad():
            latent_code = self.latent_codes(idx)
            return warp_with_latent_code(I1, latent_code, self.phi, coords, seg=I1_seg)