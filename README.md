# Underwater Scuba Sign Language Detection

A reproducible deep‐learning framework that automatically recognizes scuba‐diving hand signals in challenging underwater conditions using the CADDY dataset.  
Built in Google Colab with Fast.ai/PyTorch, it supports scenario‐aware splits, ResNet‐34/50 backbones, advanced augmentations, and exports a serialized Pickle model for direct deployment on Autonomous Underwater Vehicles (AUVs).

---

## 📖 Table of Contents

- [🚀 Features](#-features)  
- [🗄️ Dataset](#️-dataset)  
- [🗂️ Repository Structure](#️-repository-structure)  
- [⚙️ Installation](#️-installation)  
- [🛠️ Data Preparation & Splitting](#️-data-preparation--splitting)  
- [🧠 Model Architecture](#️-model-architecture)  
- [🏋️ Training Pipeline](#️-training-pipeline)  
- [🔎 Evaluation](#️-evaluation)  
- [📦 Export & Deployment](#️-deployment)  
- [🤝 Contributing](#️-contributing)  

---

## 🚀 Features

- **CADDY Underwater Dataset**  
  Uses the publicly available CADDY hand‐gesture dataset (crystal‐clear, cloudy, dark sequences).  
- **Scenario‐Aware Splits**  
  Eight individual scenarios (e.g. `biograd-A/B/C`, `genova-A`, `brodarski-A/B/C/D`) plus an aggregated “overall” split, ensuring no leakage between train/valid/test.  
- **Reproducibility**  
  Deterministic seeding (numpy, PyTorch, Python hash), MD5 checksum verification, and GPU memory monitoring via `nvidia-smi`.  
- **Modular Preprocessing**  
  Python scripts (and Colab notebook) to unzip, validate, list files, and programmatically create splits.  
- **Fast.ai Training**  
  Easy switch between ResNet‐34 and ResNet‐50; rich augmentation (rotations, zoom, lighting, warp); callbacks for live training graphs and best‐model checkpointing.  
- **Comprehensive Evaluation**  
  • Top-loss visualization  
  • Confusion matrices & most-confused class pairs  
  • Per-scenario accuracy reports  
  • Exportable CSV/pickle results  
- **Pickle Model Export**  
  Entire trained `Learner` is serialized to a `.pkl` file—load and infer in any Python environment (e.g. on an AUV).  

---

## 🗄️ Dataset

- **Name:** CADDY Underwater Hand Gesture Dataset  
- **Source:** [CNR-ISSIA CADDY](http://www.caddian.eu)  
- **Structure:**  
all-scenarios/
├─ biograd-A/
├─ biograd-B/
├─ …
└─ brodarski-D/

- **Usage:** Download the ZIP, verify with MD5 = `88dd6fbfc8176d6845dd0f55f95c0c5b`, then unzip.
---
## 🗂️ Repository Structure
├── data/ # Scripts to download & verify CADDY dataset
├── splits/ # Generated train/valid/test folders per scenario
├── notebooks/ # Google Colab notebook (main pipeline)
├── scripts/
│ ├── prepare_data.py # Unzip, MD5-check, directory listing, scenario splits
│ ├── train_model.py # Fast.ai training loop (ResNet-34/50, callbacks)
│ ├── evaluate_model.py # Interpretation & metrics (top losses, confusion mat.)
│ └── export_model.py # Serialize Learner to Pickle
├── results/
│ ├── figures/ # Plots (loss curves, confusion matrices)
│ ├── train_valid-df-model-<M>.csv # CSV of train/valid file lists
│ ├── test-df-model-<M>.csv # CSV of test file lists
│ ├── accuracies-df-model-<M>.csv # Per-scenario accuracy report
│ └── model-<M>.pkl # Serialized model for deployment
├── requirements.txt # fastai, torchvision, etc.
└── README.md # This file

---

## ⚙️ Installation

```bash
git clone https://github.com/rushabhdhoke/Underwater-scuba-sign-language-Detection.git
cd Underwater-scuba-sign-language-Detection
pip install -r requirements.txt
```

## 🧠 Model Architecture
Backbones: ResNet-34 or ResNet-50 (torchvision.models)
Head: Fast.ai default (adaptive pooling → dense → dropout → softmax)
Loss: CrossEntropyLoss
Optimizer: Adam via fit_one_cycle
Augmentations: Rotation, zoom, lighting, warp, normalization to ImageNet stats


## 🏋️ Training Pipeline
Load Data:

```python
from fastai.vision.all import *
data = ImageDataLoaders.from_folder(
    Path('splits'),
    valid_pct=0.2,
    bs=32,
    item_tfms=Resize(224),
    batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
)
```
### Create Learner:

```python
learn = cnn_learner(data, resnet50, metrics=accuracy)
```

### Train 
```python
learn.fit_one_cycle(
  10,
  lr_max=1e-3,
  cbs=[ShowGraphCallback(), SaveModelCallback(monitor='accuracy',
                                              fname='resnet50-stage1')]
)
```

### Export

```python
learn.export('results/model-F.pkl')
```


## 🔎 Evaluation
### Top-Loss Visualization:
```
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9, figsize=(15,11))
```

### Confusion Matrix & Most Confused:
```python
interp.plot_confusion_matrix(figsize=(12,12))
print(interp.most_confused(min_val=2))
```

### Per-Scenario Accuracy:
```python
learn.dls = data_test['genova-A']
preds, targs = learn.get_preds()
acc = accuracy(preds, targs)
```

## 📦 Deployment

Load model-F.pkl on your onboard Python stack; feed live frame crops to learner.predict(); trigger AUV actions based on recognized gestures.


## 🤝 Contributing
- Fork the repo
- Create a branch: git checkout -b feature/YourFeature
- Commit: git commit -am 'Add feature'
- Push: git push origin feature/YourFeature
- Open a PR
