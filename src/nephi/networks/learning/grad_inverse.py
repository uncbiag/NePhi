import torch
import torch.nn as nn

from nephi.utils import gradient, warp


class RegistrationTwoMLPGradInverse(nn.Module):
    def __init__(
        self, encoder, phi, phi_inv, sim_sampler, reg_sampler, sim, lamb, prior_lamb
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.phi = phi
        self.phi_inv = phi_inv
        self.lamb = lamb
        self.sim_sampler = sim_sampler
        self.reg_sampler = reg_sampler
        self.sim = sim
        self.prior_lamb = prior_lamb

        with torch.no_grad():
            # Init the displacement from zero
            torch.nn.init.zeros_(self.phi.synthesis_nw.final_layer.weight)
            torch.nn.init.zeros_(self.phi.synthesis_nw.final_layer.bias)

            torch.nn.init.zeros_(self.phi_inv.synthesis_nw.final_layer.weight)
            torch.nn.init.zeros_(self.phi_inv.synthesis_nw.final_layer.bias)

    def forward(self, I1, I2, idx):
        latent_code = self.encoder(I1, I2)

        sim_loss = self.compute_sim(I1, I2, latent_code)
        reg_loss = self.reg(latent_code)
        prior_loss = self.prior_reg(latent_code)
        loss = sim_loss + self.lamb * reg_loss + prior_loss * self.prior_lamb
        return loss, {"sim": sim_loss.detach(), "reg": reg_loss.detach()}

    def prior_reg(self, latent_code):
        return latent_code.reg()

    def compute_sim(self, I1, I2, latent_code):
        def _sim(phi_fn, I1, I2, latent_code):
            x = self.sim_sampler(
                batch_size=latent_code.batch_size, device=latent_code.device
            )
            phi = phi_fn(x, latent_code(x))["model_out"] + x
            for _ in range(len(I1.shape[2:]) - 1):
                phi = phi.unsqueeze(1)
                x = x.unsqueeze(1)
            warped_img = torch.nn.functional.grid_sample(
                I1, phi.flip(-1), align_corners=True
            )
            target = torch.nn.functional.grid_sample(I2, x.flip(-1), align_corners=True)
            return self.sim(warped_img, target)

        return (
            _sim(self.phi, I1, I2, latent_code)
            + _sim(self.phi_inv, I2, I1, latent_code)
        ) * 0.5

    def reg(self, latent_code):
        def _reg(phi_fn, phi_inv_fn, latent_code):
            x = self.reg_sampler(
                batch_size=latent_code.batch_size, device=latent_code.device
            ).requires_grad_()

            x_ = phi_fn(x, latent_code(x))["model_out"] + x
            y = phi_inv_fn(x_, latent_code(x_))["model_out"] + x_ - x
            dim = x.shape[2]
            return torch.mean(
                torch.stack([gradient(y[:, :, i], x) for i in range(dim)], dim=-1) ** 2
            )

        return (
            _reg(self.phi, self.phi_inv, latent_code)
            + _reg(self.phi_inv, self.phi, latent_code)
        ) * 0.5

    def register(self, I1, I2, coords, idx, I1_seg=None):
        with torch.no_grad():
            return warp(I1, I2, self.encoder, self.phi, coords, seg=I1_seg)
