
<h1 align="center">🌀 PromptCT: Prompting Lipschitz-constrained Network for Multiple-in-One Sparse-view CT Reconstruction</h1>

<p align="center">
  <strong>Baoshun Shi, Ke Jiang, Qiusheng Lian, Xinran Yu, Huazhu Fu</strong><br>
  <em>TMI 2025 (Under Review)</em>
</p>


<p align="center">
  <a href="https://github.com/shibaoshun/PromptCT">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="github">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c?logo=pytorch" alt="pytorch">
  </a>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="license">
</p>


---

> **Abstract**  
> Despite significant advancements in deep learning-based sparse-view computed tomography (SVCT) reconstruction algorithms, these methods still encounter two primary limitations: (i) It is challenging to explicitly prove that the prior networks of deep unfolding algorithms satisfy Lipschitz constraints due to their empirically designed nature. (ii) The substantial storage costs of training a separate model for each setting in the case of multiple views hinder practical clinical applications. To address these issues, we elaborate an explicitly provable Lipschitz-constrained network, dubbed LipNet, and integrate an explicit prompt module to provide discriminative knowledge of different sparse sampling settings, enabling the treatment of multiple sparse view configurations within a single model. Furthermore, we develop a storage-saving deep unfolding framework for multiple-in-one SVCT reconstruction, termed **PromptCT**, which embeds LipNet as its prior network to ensure the convergence of its corresponding iterative algorithm. In simulated and real data experiments, PromptCT outperforms benchmark reconstruction algorithms in multiple-in-one SVCT reconstruction, achieving higher-quality reconstructions with lower storage costs. On the theoretical side, we explicitly demonstrate that LipNet satisfies boundary property, further proving its Lipschitz continuity and subsequently analyzing the convergence of the proposed iterative algorithms.

---
**Network Architecture:**  
![Network Architecture](https://github.com/shibaoshun/PromptCT/blob/main/fig/fig1.jpg?raw=true)




## 📚 Table of Contents

- [🚀 Installation](#-installation)
- [📂 Dataset Preparation](#-dataset-preparation)
- [🧠 Training](#-training)
- [🔍 Testing](#-testing)
- [📈 Results](#-results)
- [📄 Citation](#-citation)
- [📜 License & Acknowledgement](#-license--acknowledgement)

---

## 🚀 Installation

This project is implemented with **PyTorch 1.10.0** and supports both **CPU** and **GPU (CUDA)** environments.  
For best performance, we recommend running on a GPU-enabled system.

### ✅ Recommended Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10.0  
- CUDA ≥ 11.3  
- GPU: NVIDIA RTX 3090 / A100 or equivalent  
- OS: Ubuntu 20.04 / Windows 10+

### 🔧 Installation Steps

### Please make sure your environment meets the following requirements:

```bash
git clone https://github.com/shibaoshun/PromptCT.git
cd PromptCT
pip install -r requirements.txt
```

## 📂 Dataset Preparation

You can download the **training and testing datasets** from Baidu Drive:

📁 **Dataset**
 🔗 https://pan.baidu.com/s/1lQRFUrkaUH7uEDB6iyKq0Q?pwd=2025
 🔑 **Password:** `2025`

After downloading, place the data under:

```
PromptCT/
 ├── data/
 │    ├── train/
 │    ├── val/
 │    └── test/
```

------

## 🧠 Training

You can train the model using the default configuration with a single command:

```
python main.py --phase tr
```

------

## 🔍 Testing

### 🔸 Pretrained Models

Download pretrained models from:

📁 **Pretrained Checkpoints**
 🔗 https://pan.baidu.com/s/1xbyzy7vOlVyc-vQjKDqcBA?pwd=2025
 🔑 **Password:** `2025`

Place them under:

```
PromptCT/
 ├── result/
 │    ├── sparse/
 │    │    ├── ckp/
 │    │    │    └── best.pth
```

### 🔸 Example Testing Command

```
python main.py --phase test
```

> Results and reconstruction outputs will be saved under `./result/`.

------

## 📈 Results

PromptCT achieves **superior reconstruction quality** with significantly **lower storage costs**, enabling **multiple sparse-view CT reconstruction in a single model**.

| Method       | PSNR (dB) ↑ | SSIM ↑ | Storage ↓ |
| ------------ | ----------- | ------ | --------- |
| FBP          | 16.17       | 0.5719 | -         |
| LipCT        | 39.37       | 0.9406 | 37.9      |
| MLipCT       | 38.62       | 0.9351 | 9.5       |
| **PromptCT** | 39.49       | 0.9411 | 12.5      |

------

## 📄 Citation

If you find this work useful, please cite:

```
@article{PromptCT2025,
  title   = {Prompting Lipschitz-constrained network for multiple-in-one sparse-view CT reconstruction},
  author  = {Baoshun Shi and Ke Jiang and Qiusheng Lian and Xinran Yu and Huazhu Fu},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2025}
}
```

------

## 📜 License & Acknowledgement

This project is released under the MIT License.
 We would like to thank all contributors and referenced works for their valuable resources and datasets.

------

<p align="center">⭐ If you find this repository useful, please give it a star to support us! ⭐</p> ```

