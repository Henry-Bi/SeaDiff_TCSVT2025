# SeaDiff: Underwater Image Enhancement with Degradation-Aware Diffusion Model

This is the official repository for the paper [SeaDiff: Underwater Image Enhancement with Degradation-Aware Diffusion Model].

# News

- 2025.06.12: The initial version of the code is uploaded.

## Environment

- python >= 3.8
- pytorch >= 1.7.0
- torchvision >= 0.8.0

## Dataset Preparation

To train SeaDiff, you should download the UIE datasets.

Then use [Depth Anything](https://github.com/DepthAnything/Depth-Anything-V2) to estimate monocular depth maps.

Third, use utils/create_hist_sample.py to estimate histogram representations.

After preprocessing, our folder structure is as follows:
```shell
datasets/
└── UIEB/
    ├── train/
    │   ├── input/
    │   ├── label/
    │   ├── depth/
    │   └── histo/
    └── val/
        ├── input/
        ├── label/
        ├── depth/
        └── histo/
```


## 🌟 Training and 🎇 Testing

Whether it's for training or inference, you just need to modify the configuration parameters in `conf.yml` and run `main.py`. MODE=1 is for training, MODE=0 is for inference.


## 📜 Citation

If you find our work useful, please cite:

## 🤝 Acknowledgements
Our code is based on [DocDiff](https://github.com/Royalvice/DocDiff), [HistoGAN](https://github.com/mahmoudnafifi/HistoGAN/tree/master) and [Depth Anything](https://github.com/jiaowoguanren0615/DepthAnythingV2). We thank the authors for their excellent work!

If you have any questions, please don't hesitate to open an issue or contact Hengyue Bi at [bihengyue@stu.ouc.edu.cn](mailto:bihengyue@stu.ouc.edu.cn). 🤞🤞🤞

