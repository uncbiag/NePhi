# NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration
[![arXiv](https://img.shields.io/badge/arXiv-2203.05565-b31b1b.svg)](https://arxiv.org/abs/2309.07322)

This is the official repository for

**NePhi: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration.** \
[Lin Tian](https://www.cs.unc.edu/~lintian/), Hastings Greer, Ra\'ul San Jos\'e Est\'epar, Roni Sengupta, Marc Niethammer\
ECCV 2024.

This work proposes NePhi, a generalizable neural deformation model which results in approximately diffeomorphic transformations. In contrast to the predominant voxel-based transformation fields used in learning-based registration approaches, NePhi represents deformations functionally, leading to great flexibility within the design space of memory consumption during training and inference, inference time, registration accuracy, as well as transformation regularity. Specifically, NePhi 1) requires less memory compared to voxel-based learning approaches, 2) improves inference speed by predicting latent codes, compared to current existing neural deformation based registration approaches that only rely on optimization, 3) improves accuracy via instance optimization, and 4) shows excellent deformation regularity which is highly desirable for medical image registration. We demonstrate the performance of NePhi on a 2D synthetic dataset as well as for real 3D medical image datasets (e.g., lungs and brains). Our results show that NePhi can match the accuracy of voxel-based representations in a single-resolution registration setting. For multi-resolution registration, our method matches the accuracy of current SOTA learning-based registration approaches with instance optimization while reducing memory requirements by a factor of five.
![Model Structure](/pages/static/images/NePhi_pipeline.png)

**This repository is under active maintenance.** Here is the todo list:
- [ ] Commit evaluation scripts
- [ ] Upload model weights to cloud drive

## Setup environment
```
git clone https://github.com/uncbiag/NePhi.git
cd NePhi
conda create -n nephi python=3.7
pip install -e .
```

## Train the single-res nephi model
```
python demos/train.py -o=[OUTPUT_FOLDER] -d=[DATASET_FOLDER] -e=[EXPERIMENT_NAME] --train_config=./demos/configs/hybrid_encoder_single_res_reg_full_lung_config.py --epochs_pretrain=2600 --eval_period=-1 --save_period=200 --batch_size=8 -g 0 1 2 3 --lr=1e-4 --lr_step=3000 --with_augmentation=0
```

## Citation
```
@article{tian2023texttt,
  title={$$\backslash$texttt $\{$NePhi$\}$ $: Neural Deformation Fields for Approximately Diffeomorphic Medical Image Registration},
  author={Tian, Lin and Greer, Hastings and Est{\'e}par, Ra{\'u}l San Jos{\'e} and Sengupta, Roni and Niethammer, Marc},
  journal={arXiv preprint arXiv:2309.07322},
  year={2023}
}
```
