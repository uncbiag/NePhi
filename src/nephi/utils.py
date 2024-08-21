import torch
import numpy as np
import os
import subprocess

def sampler(num=1000, dim=2, within_frame=False):
    if within_frame:
        return torch.rand((num, dim)) * 2.0 - 1.0
    else:
        return torch.rand((num, dim)) * 2.2 - 1.1

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def compute_encoder_output_shape(stride_list, shape, prod=True):
    from functools import reduce

    out_shape = [
        reduce(lambda x, y: (x - 1) // y + 1, [shape[i]] + stride_list)
        for i in range(len(shape))
    ]
    return out_shape if not prod else reduce(lambda x, y: x * y, out_shape)

def warp_with_latent_code(I1, latent_code, phi, coords, batch_size=10000, seg=None):
    ori_shape = coords.shape
    if ori_shape[0] != I1.shape[0]:
        # print(f"Expand coordinates to batch size{I1.shape}")
        coords = coords.expand(I1.shape[0], *[-1] * len(ori_shape[1:]))
        ori_shape = coords.shape
    coords = coords.view(ori_shape[0], -1, ori_shape[-1])

    # Evaluate with batch size.
    phis = []
    for i in range(int((coords.shape[1] - 1) / batch_size) + 1):
        x = coords[:, i * batch_size : (i + 1) * batch_size]
        # phis.append(phi(x, latent_code(x))["model_out"])
        phis.append(phi(x, latent_code(x))["model_out"] + x)

    transform = (torch.cat(phis, dim=1)).reshape(*ori_shape)
    del phis, x

    if ori_shape[-1] == 2:
        return (
            torch.nn.functional.grid_sample(I1, transform.flip(-1), align_corners=True),
            transform.permute(0, 3, 1, 2),
            torch.nn.functional.grid_sample(
                seg, transform.flip(-1), align_corners=True, mode="nearest"
            )
            if seg is not None
            else None,
        )
    else:
        return (
            torch.nn.functional.grid_sample(I1, transform.flip(-1), align_corners=True),
            transform.permute(0, 4, 1, 2, 3),
            torch.nn.functional.grid_sample(
                seg, transform.flip(-1), align_corners=True, mode="nearest"
            )
            if seg is not None
            else None,
        )
    
def warp(I1, I2, encoder, phi, coords, batch_size=10000, seg=None):
    latent_code_as_function = encoder(I1, I2)
    ori_shape = coords.shape
    if ori_shape[0] != I1.shape[0]:
        print(f"Expand coordinates to batch size{I1.shape}")
        coords = coords.expand(I1.shape[0], *[-1] * len(ori_shape[1:]))
        ori_shape = coords.shape
    coords = coords.view(ori_shape[0], -1, ori_shape[-1])

    # Evaluate with batch size.
    phis = []
    for i in range(int((coords.shape[1] - 1) / batch_size) + 1):
        x = coords[:, i * batch_size : (i + 1) * batch_size]
        phis.append(phi(x, latent_code_as_function(x))["model_out"] + x)

    transform = (torch.cat(phis, dim=1)).reshape(*ori_shape)
    del phis, x

    if ori_shape[-1] == 2:
        return (
            torch.nn.functional.grid_sample(I1, transform.flip(-1), align_corners=True),
            transform.permute(0, 3, 1, 2),
            torch.nn.functional.grid_sample(
                seg, transform.flip(-1), align_corners=True, mode="nearest"
            )
            if seg is not None
            else None,
        )
    else:
        return (
            torch.nn.functional.grid_sample(I1, transform.flip(-1), align_corners=True),
            transform.permute(0, 4, 1, 2, 3),
            torch.nn.functional.grid_sample(
                seg, transform.flip(-1), align_corners=True, mode="nearest"
            )
            if seg is not None
            else None,
        )

def get_mgrid(shape, flatten=True):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int"""
    tensors = tuple([torch.linspace(-1, 1, steps=i) for i in shape])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing="ij"), dim=-1)
    if flatten:
        mgrid = mgrid.reshape(-1, len(shape))
    return mgrid

def flips(phi, in_percentage=False):
    if len(phi.size()) == 5:
        a = (phi[:, :, 1:, 1:, 1:] - phi[:, :, :-1, 1:, 1:]).detach()
        b = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, :-1, 1:]).detach()
        c = (phi[:, :, 1:, 1:, 1:] - phi[:, :, 1:, 1:, :-1]).detach()

        dV = torch.sum(torch.cross(a, b, 1) * c, axis=1, keepdims=True)
        if in_percentage:
            return torch.mean((dV < 0).float()) * 100.0, dV
        else:
            return torch.sum(dV < 0) / phi.shape[0], dV
    elif len(phi.size()) == 4:
        du = (phi[:, :, 1:, :-1] - phi[:, :, :-1, :-1]).detach()
        dv = (phi[:, :, :-1, 1:] - phi[:, :, :-1, :-1]).detach()
        dA = du[:, 0] * dv[:, 1] - du[:, 1] * dv[:, 0]
        if in_percentage:
            return torch.mean((dA < 0).float()) * 100.0, dA
        else:
            return torch.sum(dA < 0) / phi.shape[0], dA
    elif len(phi.size()) == 3:
        du = (phi[:, :, 1:] - phi[:, :, :-1]).detach()
        if in_percentage:
            return torch.mean((du < 0).float()) * 100.0, du
        else:
            return torch.sum(du < 0) / phi.shape[0], du
    else:
        raise ValueError()

def dice(seg_A, seg_B):
    assert seg_A.shape[1] == 1
    assert seg_B.shape[1] == 1

    if len(seg_A.shape) == 4:
        reduce_dim = [1, 2, 3]
    elif len(seg_A.shape) == 5:
        reduce_dim = [1, 2, 3, 4]
    else:
        print("The length of dimension of image is not supported.")
        return None

    len_intersection = torch.sum(seg_A * seg_B, reduce_dim)
    fn = torch.sum(seg_B, reduce_dim) - len_intersection
    fp = torch.sum(seg_A, reduce_dim) - len_intersection

    return 2 * len_intersection / (2 * len_intersection + fn + fp + 1e-10)


def dice_multi(seg_A, seg_B, labels=None):
    if labels == None:
        labels = torch.unique(torch.cat([seg_A, seg_B]))

    if len(seg_A.shape) == 4:
        reduce_dim = [1, 2, 3]
    elif len(seg_A.shape) == 5:
        reduce_dim = [1, 2, 3, 4]
    else:
        print("The length of dimension of image is not supported.")
        return None

    dices = []
    for l in labels:
        if l == 0:
            # Skip background
            continue
        pred_i = seg_A == l
        true_i = seg_B == l
        len_intersection = torch.sum(pred_i * true_i, reduce_dim)
        dices.append(
            2
            * len_intersection
            / (torch.sum(pred_i, reduce_dim) + torch.sum(true_i, reduce_dim) + 1e-10)
        )

    return torch.mean(torch.stack(dices, dim=1))

def ncc(image_A, image_B):
    A = normalize(image_A[:, :1])
    B = normalize(image_B)
    res = torch.mean(A * B)
    return 1 - res

def set_seed_for_demo():
    """reproduce the training demo"""
    import random

    seed = 2021
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_git_revisions_hash():
    hashes = []
    hashes.append(
        subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("ascii")
    )
    return hashes

def path_import(absolute_path):
    """implementation taken from https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly"""
    import importlib

    spec = importlib.util.spec_from_file_location(absolute_path, absolute_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module