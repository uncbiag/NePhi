import numpy as np
from icon_registration.losses import NCC
from torch.utils.data import DataLoader

from nephi.models_bak import LocalMLP
from nephi.networks import RegistrationTwoMLPGradInverse
from nephi.modules import (
    GlobalAndLocalEncoder,
    RandomSampler,
)
import math

data_shape = (175, 175, 175)


class train_config:
    def __init__(self):
        pass

    def make_pretrain_net(self):
        default_setting = {
            "in_features": 3,
            "out_features": 3,
            "hidden_layers": 2,
            "hidden_features": 128,
            "latent_dim": 64,
            "synthesis_activation": "sine",
            "modulation_activation": "sine",
            "concat": True,
            "embedding": "ffn",
            "N_freqs": 128,
        }

        local_feature_layer_num = round(math.log2(max(data_shape))) - 4 - 1
        enc_filters = [8, 16, 32, 64, 128, 256] + [256] * max(
            0, (round(math.log2(max(data_shape))) - 7)
        )
        sim_sample_num = min(100000, int(0.02 * np.prod(data_shape)))
        reg_sample_num = min(500, max(10, int(0.0001 * np.prod(data_shape))))
        print(f"local_feature_layer_num: {local_feature_layer_num}")
        print(f"enc_filter: {enc_filters}")
        net = RegistrationTwoMLPGradInverse(
            GlobalAndLocalEncoder(
                latent_dim=default_setting["latent_dim"] // 2,
                data_shape=data_shape,
                enc_filters=enc_filters,
                local_feature_layer_num=local_feature_layer_num,
            ),
            LocalMLP(**default_setting),
            LocalMLP(**default_setting),
            sim_sampler=RandomSampler(sim_sample_num, dim=3),
            reg_sampler=RandomSampler(reg_sample_num, dim=3),
            sim=NCC(),
            lamb=10.0,
            prior_lamb=0.01,
        )
        return net

    def make_dataset(self, data_path, batch_size, phases=["train", "test"]):
        train_loader, test_loader = None, None

        if "train" in phases:
            train_loader = DataLoader(
                self.get_dataset(
                    data_path,
                    phase="train",
                ),
                batch_size=batch_size,
                shuffle=True,
                drop_last=True,
            )

        if "test" in phases:
            test_loader = DataLoader(
                self.get_dataset(
                    data_path,
                    phase="test",
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True,
            )
        return train_loader, test_loader, None

    def get_dataset(
        self,
        dataset_folder,
        phase="train",
    ):
        """
        Return dataloader from the dataset_folder
        """
        from nephi.dataset import COPDProcessedDataset

        dataset = COPDProcessedDataset(phase, data_path=dataset_folder, with_seg=True)

        return dataset
