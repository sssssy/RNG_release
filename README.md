# [CVPR 2025] RNG: Relightable Neural Gaussians

This repository is the official code for CVPR 2025 paper *RNG: Relightable Neural Gaussians*.

Please refer to the [project page](https://whois-jiahui.fun/project_pages/RNG/index.html) for more information.

## Dataset

This code should be compatible with the dataset structure from [NRHints](https://github.com/iamNCJ/NRHints).

## Usage

### Setup

```bash
$ cd ./submodules/diff-gaussian-rasterization
$ python setup.py install
```

### Train

Example:

```bash
## stage 1 (forward-shading)
$ python train.py -s DATA_PATH -m OUTPUT_PATH --iterations 30000 --save_iteration 7000 30000 --densify_until_iter 30000 --eval --json --color_mlp --in_channels 16 --max_training_images 1000 --max_reso 512 --loss l1

## stage 2 (deferred-shading)
$ python train.py -s DATA_PATH -m OUTPUT_PATH --iterations 100000 --save_iteration 100000  --load_pc OUTPUT_PATH/chkpnt30000.pth  --eval --json --color_mlp --defer_shading --in_channels 16 --max_training_images 1000 --max_reso 512 --shadow_map --shadow_grad --depth_mlp --depth_mlp_modifier 1.0 --encoding_levels_each 2 --encoding_levels_shadow 8 --crop_pc 1.0 --loss l1
```


### Render

Example:

```bash
$ python render.py -m OUTPUT_PATH  --iteration 100000 --eval --json --color_mlp --defer_shading --in_channels 16 --shadow_map --depth_mlp --depth_mlp_modifier 1.0 --encoding_levels_each 2 --encoding_levels_shadow 8 --max_reso 512 --crop_pc 1.0 [--output_depth --output_alpha --output_shadow]
```

## BibTeX
```
@article{
  author={Jiahui Fan and Fujun Luan and Jian Yang and Milos Hasan and Beibei Wang},
  title={RNG: Relightable Neural Gaussians},
  year={2025},
  journal={Proceedings of CVPR 2025},
}
```
