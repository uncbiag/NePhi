import torch
from tqdm import tqdm
from nephi.utils import get_mgrid, flips, dice_multi, ncc
import skimage
from skimage.util import compare_images
import numpy as np
import torch.nn.functional as F
import random
from collections import defaultdict

def write_scalars(scalar_dict, writer, epoch, pre_fix=""):
    for k, v in scalar_dict.items():
        writer.add_scalar(pre_fix + k, v, epoch)


def write_imgs(img_dict, writer, epoch):
    for k, v in img_dict.items():
        writer.add_images(k, v, epoch)


def save_model(net, opt, epoch, pre_fix, output_folder, lr_scheduler=None):
    if not pre_fix == "":
        pre_fix += "_"

    if hasattr(net, "phi_inv") and hasattr(net, "encoder"):
        torch.save(
            {
                "encoder": net.encoder.state_dict(),
                "phi": net.phi.state_dict(),
                "phi_inv": net.phi_inv.state_dict(),
                "epochs": epoch,
            },
            f"{output_folder}/checkpoints/{pre_fix}net_{epoch:05}",
        )
    elif hasattr(net, "encoder"):
        torch.save(
            {
                "encoder": net.encoder.state_dict(),
                "phi": net.phi.state_dict(),
                "epochs": epoch,
            },
            f"{output_folder}/checkpoints/{pre_fix}net_{epoch:05}",
        )
    else:
        torch.save(
            {
                "net": net.state_dict(),
                "epochs": epoch,
            },
            f"{output_folder}/checkpoints/{pre_fix}net_{epoch:05}",
        )

    opt_state_dict = {"optimizer": opt.state_dict(), "epochs": epoch}
    if lr_scheduler is not None:
        opt_state_dict["lr_scheduler"] = lr_scheduler.state_dict()
    torch.save(
        opt_state_dict,
        f"{output_folder}/checkpoints/{pre_fix}opt_{epoch:05}",
    )

def train_kenel_with_idx(net, train_loader, opt, device, with_augmentation=False):
    running_loss = defaultdict(lambda: 0.0)
    net.train()
    for I1, I2, idx in train_loader:
        opt.zero_grad()

        I1, I2 = I1[:, :1].to(device), I2[:, :1].to(device)
        if with_augmentation:
            I1, I2 = apply_augmentation(I1, I2, device)
        loss, logs = net(I1, I2, idx)

        loss = torch.mean(loss)
        loss.backward()
        opt.step()

        for k, v in logs.items():
            running_loss[k] += torch.mean(v)

    for k, v in running_loss.items():
        running_loss[k] /= len(train_loader)
        running_loss[k] = running_loss[k].item()
    return running_loss


def eval_kernel_with_idx(net, val_loader, coords, device, with_plot=False):
    running_loss = {
        "dice": 0.0,
        "sim": 0.0,
        "fold": 0.0,
    }
    net.eval()
    coords = coords.to(device)
    for I1, I2, idx in val_loader:
        I1, I2, I1_seg, I2_seg = (
            I1[:, :1].to(device),
            I2[:, :1].to(device),
            I1[:, 1:2].to(device),
            I2[:, 1:2].to(device),
        )
        warped_img, transformation, warped_seg = net.register(
            I1, I2, coords, idx, I1_seg
        )
        sim_loss = ncc(warped_img, I2)
        fold, fold_m = flips(transformation, in_percentage=True)

        running_loss["sim"] += sim_loss.item()
        running_loss["dice"] += dice_multi(warped_seg, I2_seg).mean().item()
        running_loss["fold"] += fold.item()

    # If we need to return the figures on the last iteration
    to_plot = None
    if with_plot:
        shape = I1.shape
        if len(shape) == 5:
            warped_img_slice = warped_img.detach()[:, :, :, shape[3] // 2].cpu()
            target_slice = I2[:, :, :, shape[3] // 2].cpu()
            moving_slice = I1[:, [0, 0, 0], :, shape[3] // 2].cpu()
        elif len(shape) == 4:
            warped_img_slice = warped_img.detach().cpu()
            target_slice = I2.cpu()
            moving_slice = I1.cpu()
        to_plot = {
            "moving": moving_slice,
            "target": target_slice[:, [0, 0, 0]],
            "warped": warped_img_slice[:, [0, 0, 0]],
            "difference": torch.from_numpy(
                np.array(
                    [
                        compare_images(
                            x[0],
                            y[0],
                            method="checkerboard",
                        )
                        for x, y in zip(warped_img_slice.numpy(), target_slice.numpy())
                    ]
                )
            ).unsqueeze(1)[:, [0, 0, 0]],
        }

    return {
        "dice": running_loss["dice"] / len(val_loader),
        "sim": running_loss["sim"] / len(val_loader),
        "fold": running_loss["fold"] / len(val_loader),
    }, to_plot

def apply_augmentation(image_A, image_B, device):
    with torch.no_grad():
        identity_list = []
        for i in range(image_A.shape[0]):
            identity = torch.Tensor([[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]])
            idxs = set((0, 1, 2))
            for j in range(3):
                k = random.choice(list(idxs))
                idxs.remove(k)
                identity[0, j, k] = 1
            identity = identity * (torch.randint_like(identity, 0, 2) * 2 - 1)
            identity_list.append(identity)

        identity = torch.cat(identity_list)

        noise = torch.randn((image_A.shape[0], 3, 4))

        forward = identity + 0.05 * noise

        grid_shape = list(image_A.shape)
        grid_shape[1] = 3
        forward_grid = F.affine_grid(forward.to(device), grid_shape, align_corners=True)

        warped_A = F.grid_sample(
            image_A, forward_grid, padding_mode="border", align_corners=True
        )

        noise = torch.randn((image_A.shape[0], 3, 4))
        forward = identity + 0.05 * noise

        grid_shape = list(image_A.shape)
        grid_shape[1] = 3
        forward_grid = F.affine_grid(forward.to(device), grid_shape, align_corners=True)
        warped_B = F.grid_sample(
            image_B, forward_grid, padding_mode="border", align_corners=True
        )

    return (warped_A, warped_B)  # if random.random() > 0.5 else (warped_B, warped_A)

def train_with_tensorboard(
    net,
    opt,
    dataset,
    total_epochs,
    device,
    eval_period,
    save_period,
    output_folder,
    writer=None,
    start_epoch=0,
    with_augmentation=False,
    lr_scheduler=None,
):
    train_loader, val_loader, dataset_sampler = dataset

    sample_data = next(iter(train_loader))[0]
    data_shape = sample_data.shape[2:]
    coords = get_mgrid(data_shape).unsqueeze(0).view(1, *data_shape, len(data_shape))
    coords = coords.expand(sample_data.shape[0], *[-1] * len(coords.shape[1:]))

    assert (
        eval_period > 0 and writer is not None
    ) or eval_period <= 0, "When evaluation is enabled, writer must be provided."

    for epoch in tqdm(range(start_epoch, total_epochs)):
        if dataset_sampler is not None:
            dataset_sampler.set_epoch(epoch)

        # Eval
        if eval_period > 0 and epoch % eval_period == 0:
            with torch.no_grad():
                if isinstance(net, torch.nn.DataParallel):
                    net_inner = net.module
                else:
                    net_inner = net
                net_inner.eval()
                log, to_plot = eval_kernel_with_idx(
                    net_inner, val_loader, coords, device, with_plot=True
                )
                write_scalars(log, writer, epoch, pre_fix="val/")
                write_imgs(to_plot, writer, epoch)

                # Clean up
                net_inner = None
                del log, to_plot

        # Save model
        if save_period > 0 and epoch % save_period == 0:
            if isinstance(net, torch.nn.DataParallel):
                net_inner = net.module
            else:
                net_inner = net
            save_model(net_inner, opt, epoch, "pretrain", output_folder, lr_scheduler)
            # Clean up
            net_inner = None

        # Train
        net.train()
        log = train_kenel_with_idx(net, train_loader, opt, device, with_augmentation)
        if lr_scheduler is not None:
            log["lr"] = lr_scheduler.get_last_lr()[0]
            lr_scheduler.step()
        else:
            log["lr"] = opt.param_groups[0]["lr"]
        if writer is not None:
            write_scalars(log, writer, epoch, pre_fix="train/")
        del log

    print("Validate after the last epoch.")
    if eval_period > 0:
        with torch.no_grad():
            if isinstance(net, torch.nn.DataParallel):
                net_inner = net.module
            else:
                net_inner = net
            net_inner.eval()
            log, to_plot = eval_kernel_with_idx(
                net_inner, val_loader, coords, device, with_plot=True
            )
            write_scalars(log, writer, epoch, pre_fix="val/")
            write_imgs(to_plot, writer, epoch)

    # Save model
    if save_period > 0:
        if isinstance(net, torch.nn.DataParallel):
            net_inner = net.module
        else:
            net_inner = net
        save_model(net_inner, opt, epoch, "pretrain", output_folder, lr_scheduler)