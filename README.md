# SeaDiff

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

This is the official repository for the paper 📄 **"SeaDiff: Underwater Image Enhancement with Degradation-Aware Diffusion Model"**.

## 🔥 News
- **2025.06.15**: The initial version of the code is uploaded.


## 🛠️ Environment Setup

### Prerequisites
- Python = 3.9
- PyTorch = 2.0.0
- torchvision = 0.15.1
- CUDA = 11.7

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/SeaDiff.git
cd SeaDiff

# Create conda environment
conda create -n seadiff python=3.8
conda activate seadiff
```

## 📂 Dataset Preparation

To train SeaDiff, please follow these steps:

1. **Download UIE datasets**: 
   - [UIEB](https://li-chongyi.github.io/proj_benchmark.html)
   - [EUVP](http://irvlab.cs.umn.edu/resources/euvp-dataset)
   - [SUIM-E](https://github.com/trentqq/SUIM-E)

2. **Generate depth maps** using [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2):

3. **Create histogram representations**:
   ```bash
   python utils/create_hist_sample.py --input_dir datasets/UIEB/train/input --output_dir datasets/UIEB/train/histo
   ```

### Dataset Structure
After preprocessing, organize your data as follows:
```
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

## 🚀 Quick Start

### Training or Testing
1. Modify the configuration in `conf.yml`:
   ```yaml
   MODE: 1                    # 1 for training, 0 for inference
   PRE_ORI: 'True'            # True for $x_0$, False for $\epsilon$
   # ... other parameters
   ```

2. Start:
   ```bash
   python main.py --conf conf.yml
   ```

## 🏗️ Model Architecture

<div align="center">
<img src="assets/architecture.png" width="600"/>
<p><em>Overview of SeaDiff architecture</em></p>
</div>

## 📜 Citation

If you find our work useful, please cite:


## 🤝 Acknowledgements

Our code is based on the following excellent works:
- [DocDiff](https://github.com/Royalvice/DocDiff)
- [HistoGAN](https://github.com/mahmoudnafifi/HistoGAN/tree/master) 
- [Depth Anything V2](https://github.com/jiaowoguanren0615/DepthAnythingV2)

We thank the authors for their outstanding contributions! 🙏

## 📧 Contact

If you have any questions, please feel free to:
- 📧 Email: [bihengyue@stu.ouc.edu.cn](mailto:bihengyue@stu.ouc.edu.cn)
- 🐛 Open an [Issue](https://github.com/Henry-Bi/SeaDiff/issues)
- 💬 Start a [Discussion](https://github.com/Henry-Bi/SeaDiff/discussions)

---

<div align="center">
⭐ If you find this project helpful, please consider giving it a star! ⭐
</div>
