# CrossTooth_CVPR2025
This is the official repository of **3D Dental Model Segmentation with Geometrical Boundary Preserving**

## Requirements

This project is encouraged to run on Ubuntu 20.04 LTS or higher version.

Please follow the `requirements.txt` to install packages.

Please install `pointops` from official repository *[Point Transformer](https://github.com/POSTECH-CVLab/point-transformer)*.

## Usage

Please run the following command. You can specify the intra-oral scan model and the save path.

```commandline
python predict.py --case "some.ply" --save_path "some_mask.ply"
```

Pretrained model is at `models/PTv1/point_best_model.pth`.

We provide an intra-oral scan case `YBSESUN6_upper.ply`

## Notice

This repository contains predict script for intra-oral scan segmentation with only point features.

Competing methods are at `compete` folder.

Multi-view rendering and other scripts are at `prepare_data` folder.

More details will be released later.