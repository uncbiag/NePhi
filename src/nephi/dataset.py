import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Fv
import tqdm
import itk
import glob
from torchvision.datasets import MNIST
import pickle
import random

COPDGENE_SPLIT = "/playpen-raid2/lin.tian/projects/icon_lung/ICON_lung/splits"
COPDGENE_DATA = "/playpen-raid2/Data/Lung_Registration_transposed/"
COPDGENE_PROCESSED_DATA = "/playpen-ssd/tgreer/ICON_lung/results/half_res_preprocessed_transposed_SI"

# This dataset is borrowed from ICON repo: https://github.com/uncbiag/ICON
class TriangleDataset(torch.utils.data.Dataset):
    def __init__(self, data_size, samples, hollow) -> None:
        super().__init__()
        x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
        x = np.reshape(x, (1, data_size, data_size))
        y = np.reshape(y, (1, data_size, data_size))
        cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
        cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
        r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
        theta = np.random.random((samples, 1, 1)) * np.pi * 2
        isTriangle = np.random.random((samples, 1, 1)) > 0.5

        triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(
            np.pi / 3
        ) / np.cos((np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3)

        triangles = np.tanh(-40 * triangles)

        circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
        if hollow:
            triangles = 1 - triangles**2
            circles = 1 - circles**2

            self.images = isTriangle * triangles + (1 - isTriangle) * circles
            self.images = np.expand_dims(self.images, axis=1).astype(np.single)
        else:
            self.images = isTriangle * triangles + (1 - isTriangle) * circles
            self.images = (
                np.expand_dims(self.images, axis=1).astype(np.single) + 1.0
            ) / 2.0
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        idx = np.random.randint(0, self.samples, size=2)
        image_a = self.images[idx[0]]
        image_b = self.images[idx[1]]
        return np.concatenate([image_a, image_a > 0.2], axis=0), np.concatenate(
            [image_b, image_b > 0.2], axis=0
        )


class TriangleDatasetStatic(torch.utils.data.Dataset):
    def __init__(
        self, data_size, samples, hollow, total_num=1000, images=None, transforms=None
    ) -> None:
        super().__init__()
        if images is None:
            x, y = np.mgrid[0 : 1 : data_size * 1j, 0 : 1 : data_size * 1j]
            x = np.reshape(x, (1, data_size, data_size))
            y = np.reshape(y, (1, data_size, data_size))
            cx = np.random.random((samples, 1, 1)) * 0.3 + 0.4
            cy = np.random.random((samples, 1, 1)) * 0.3 + 0.4
            r = np.random.random((samples, 1, 1)) * 0.2 + 0.2
            theta = np.random.random((samples, 1, 1)) * np.pi * 2
            isTriangle = np.random.random((samples, 1, 1)) > 0.5

            triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(
                np.pi / 3
            ) / np.cos(
                (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
            )

            triangles = np.tanh(-40 * triangles)

            circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))
            if hollow:
                triangles = 1 - triangles**2
                circles = 1 - circles**2

                self.images = isTriangle * triangles + (1 - isTriangle) * circles
                self.images = np.expand_dims(self.images, axis=1).astype(np.single)
            else:
                self.images = isTriangle * triangles + (1 - isTriangle) * circles
                self.images = (
                    np.expand_dims(self.images, axis=1).astype(np.single) + 1.0
                ) / 2.0
        else:
            self.images = images
        self.samples = len(self.images)
        self.pair_list = np.random.randint(0, self.samples, size=(total_num, 2))
        self.transforms = transforms

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        idx = self.pair_list[index]
        image_a = self.images[idx[0]]
        image_b = self.images[idx[1]]
        if self.transforms is not None:
            image_a = self.transforms(image_a)
            image_b = self.transforms(image_b)
        return (
            np.concatenate([image_a, image_a > 0.2], axis=0),
            np.concatenate([image_b, image_b > 0.2], axis=0),
            index,
        )


class MnistDataset(torch.utils.data.Dataset):
    def __init__(self, num_per_digit=100) -> None:
        super().__init__()
        self.mnist = MNIST("./cache/", train=True, download=True)

        self.pair_list = None
        for i in range(10):
            idx = np.nonzero(np.array(self.mnist.targets == i))[0]
            if self.pair_list is not None:
                self.pair_list = np.concatenate(
                    (self.pair_list, np.random.choice(idx, size=(num_per_digit, 2))),
                    axis=0,
                )
            else:
                self.pair_list = np.random.choice(idx, size=(num_per_digit, 2))

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, index):
        idx = self.pair_list[index]
        image_a = np.array(self.mnist[idx[0]][0])
        image_b = np.array(self.mnist[idx[1]][0])
        mask_a = image_a > 0
        mask_b = image_b > 0
        image_a = image_a.astype(np.float32) / 255.0
        image_b = image_b.astype(np.float32) / 255.0
        return (
            np.stack([image_a, mask_a], axis=0),
            np.stack([image_b, mask_b], axis=0),
            index,
        )


class RetinaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_shape,
        extra_deformation=False,
        blur_sigma=None,
        warps_per_pair=20,
        fixed_vertical_offset=None,
        include_boundary=False,
        cache_folder=".",
    ) -> None:
        super().__init__()
        try:
            import elasticdeform
            import hub
        except:
            raise Exception(
                """the retina dataset requires the dependencies hub and elasticdeform.
                Try pip install hub elasticdeform"""
            )

        ds_name = f"{cache_folder}/retina{extra_deformation}{data_shape[0]}{blur_sigma}{warps_per_pair}{fixed_vertical_offset}{include_boundary}.trch"

        import os

        if os.path.exists(ds_name):
            augmented_ds1_tensor, augmented_ds2_tensor = torch.load(ds_name)
        else:
            res = []
            for batch in hub.load("hub://activeloop/drive-train").pytorch(
                num_workers=0, batch_size=4, shuffle=False
            ):
                if include_boundary:
                    res.append(batch["manual_masks/mask"] ^ batch["masks/mask"])
                else:
                    res.append(batch["manual_masks/mask"])
            res = torch.cat(res)
            ds_tensor = res[:, None, :, :, 0] * -1.0 + (not include_boundary)

            if fixed_vertical_offset is not None:
                ds2_tensor = torch.cat(
                    [torch.zeros(20, 1, fixed_vertical_offset, 565), ds_tensor],
                    axis=2,
                )
                ds1_tensor = torch.cat(
                    [ds_tensor, torch.zeros(20, 1, fixed_vertical_offset, 565)],
                    axis=2,
                )
            else:
                ds2_tensor = ds_tensor
                ds1_tensor = ds_tensor

            warped_tensors = []
            print("warping images to generate dataset")
            for _ in tqdm.tqdm(range(warps_per_pair)):
                ds_2_list = []
                for el in ds2_tensor:
                    case = el[0]
                    # TODO implement random warping on gpu
                    case_warped = np.array(case)
                    if extra_deformation:
                        case_warped = elasticdeform.deform_random_grid(
                            case_warped, sigma=60, points=3
                        )
                    case_warped = elasticdeform.deform_random_grid(
                        case_warped, sigma=25, points=3
                    )

                    case_warped = elasticdeform.deform_random_grid(
                        case_warped, sigma=12, points=6
                    )
                    ds_2_list.append(torch.tensor(case_warped)[None, None, :, :])
                    ds_2_tensor = torch.cat(ds_2_list)
                warped_tensors.append(ds_2_tensor)

            augmented_ds2_tensor = torch.cat(warped_tensors)
            augmented_ds1_tensor = torch.cat(
                [ds1_tensor for _ in range(warps_per_pair)]
            )

            torch.save((augmented_ds1_tensor, augmented_ds2_tensor), ds_name)

        if blur_sigma is None:
            augmented_ds1_tensor = F.interpolate(
                augmented_ds1_tensor, data_shape, mode="area"
            )
            augmented_ds2_tensor = F.interpolate(
                augmented_ds2_tensor, data_shape, mode="area"
            )
        else:
            augmented_ds1_tensor = Fv.gaussian_blur(
                F.interpolate(augmented_ds1_tensor, data_shape, mode="area"),
                4 * blur_sigma + 1,
                blur_sigma,
            )
            augmented_ds2_tensor = Fv.gaussian_blur(
                F.interpolate(augmented_ds2_tensor, data_shape, mode="area"),
                4 * blur_sigma + 1,
                blur_sigma,
            )

        self.img1 = augmented_ds1_tensor
        self.img2 = augmented_ds2_tensor

    def __len__(self):
        return len(self.img1)

    def __getitem__(self, index):
        image_a = self.img1[index]
        image_b = self.img2[index]
        return np.concatenate([image_a, image_a > 0.2], axis=0), np.concatenate(
            [image_b, image_b > 0.2], axis=0
        )


class COPDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        ids_file=COPDGENE_SPLIT,
        data_path=COPDGENE_DATA,
        data_num=-1,
        desire_shape=None,
        static=True,
    ):
        with open(f"{ids_file}/{phase}.txt") as f:
            self.pair_paths = f.readlines()
            self.pair_paths = list(map(lambda x: x[:-1], self.pair_paths))
        print(f"{ids_file}/{phase}.txt")
        self.data_path = data_path + "/"
        self.data_num = data_num
        self.desire_shape = desire_shape
        self.static = static

    def __len__(self):
        return len(self.pair_paths) if self.data_num < 0 else self.data_num

    def process(self, iA, isSeg=False):
        iA = iA[None, None, :, :, :]
        # SI flip
        iA = torch.flip(iA, dims=(2,))
        if isSeg:
            iA = iA.float()
            iA[iA > 0] = 1
            if self.desire_shape is not None:
                iA = F.interpolate(iA, size=self.desire_shape, mode="nearest")
        else:
            iA = iA.float()
            iA = torch.clip(iA, -1000, 0) + 1000.0
            iA = iA / 1000.0
            if self.desire_shape is not None:
                iA = F.interpolate(
                    iA,
                    size=self.desire_shape,
                    mode="trilinear",
                    align_corners=True,
                )
        return iA

    def __getitem__(self, idx):
        case_id = self.pair_paths[idx]
        image_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )
        image_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_img.nii.gz"
                        )[0]
                    )
                )
            )
        )

        seg_insp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_INSP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )
        seg_exp = self.process(
            torch.tensor(
                np.asarray(
                    itk.imread(
                        glob.glob(
                            self.data_path
                            + case_id
                            + "/"
                            + case_id
                            + "_EXP_STD*_COPD_label.nii.gz"
                        )[0]
                    )
                )
            ),
            isSeg=True,
        )

        if self.static:
            return (
                torch.cat([image_insp * seg_insp, seg_insp], dim=1)[0],
                torch.cat([image_exp * seg_exp, seg_exp], dim=1)[0],
                idx,
            )
        else:
            return (
                torch.cat([image_insp * seg_insp, seg_insp], dim=1)[0],
                torch.cat([image_exp * seg_exp, seg_exp], dim=1)[0],
            )
    
class COPDProcessedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        phase="train",
        scale="2xdown",
        data_path=COPDGENE_PROCESSED_DATA,
        seg_roi=True,
        with_seg=False,
        data_num=-1,
    ):
        if phase == "debug":
            phase = "train"
        self.imgs = torch.load(
            f"{data_path}/lungs_{phase}_{scale}_scaled", map_location="cpu"
        )
        if data_num <= 0:
            self.data_num = len(self.imgs)
        else:
            self.data_num = data_num
        self.imgs = self.imgs[: self.data_num]

        if with_seg or seg_roi:
            self.segs = torch.load(
                f"{data_path}/lungs_seg_{phase}_{scale}_scaled", map_location="cpu"
            )[: self.data_num]

            if seg_roi:
                # apply segmentation map to image
                for i, (img, seg) in enumerate(zip(self.imgs, self.segs)):
                    self.imgs[i] = (
                        ((img[0] + 1) * seg[0]).float(),
                        ((img[1] + 1) * seg[1]).float(),
                    )

            if not with_seg:
                self.segs = None
        else:
            self.segs = None

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        img_a, img_b = self.imgs[idx]
        if self.segs is not None:
            mask_a, mask_b = self.segs[idx]
            img_a = torch.cat([img_a, mask_a.float()], dim=1)
            img_b = torch.cat([img_b, mask_b.float()], dim=1)
        return img_a[0], img_b[0], idx