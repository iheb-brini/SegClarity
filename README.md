# SegClarity  
**Semantic Segmentation & Explainable AI Framework for Documents and Urban Scenes**

[![Python](https://img.shields.io/badge/Python-3.10--3.12-blue)]()  
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.6-orange)]()  
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()  
[![Paper](https://img.shields.io/badge/MDPI-Published-purple)](https://www.mdpi.com/2313-433X/11/12/424)

SegClarity is a unified framework designed for **semantic segmentation** and **explainable AI**, supporting both:
- **Document Layout Analysis**
- **Urban Scene Understanding (Cityscapes)**

It provides pre-trained models, attribution methods, visualization utilities, and experiment notebooks.

---

## ✨ Features

### 🗂 Document Segmentation
- Models trained on **UTP** and **splitAB1** datasets  
- Layout segmentation using UNet & LUNet architectures  

### 🏙 Urban Scene Segmentation
- UNet models trained on **Cityscapes**  
- Full semantic segmentation pipeline  

### 🔍 Explainable AI
- Attribution methods via **Captum**  
- Integrated Gradients, GradCAM, Occlusion, and more  
- Visual explainability on documents & scenes  

### 📊 Visualization Tools
- Side-by-side predictions  
- Attribution heatmaps  
- Overlay masks, saliency, and class‑wise contributions  

---

## 📁 Project Structure

```
SegClarity/
├── Modules/
│   ├── Architecture/         # UNet, LUNet implementations
│   ├── Dataset/              # Dataset loaders & transforms
│   ├── CityscapeDataset/     # Cityscapes utilities
│   ├── ModelXAI/             # Explainable AI methods
│   ├── Attribution/          # Attribution pipeline
│   ├── Visualization/        # Plotting & rendering utils
│   └── ...
├── Notebooks/                # Experiment notebooks
├── models/                   # Pre-trained weights
├── datasets/                 # Document datasets
└── requirements.txt
```

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/iheb-brini/SegClarity.git
cd SegClarity
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

📌 *Note: Requirements include PyTorch with CUDA 12.6. Adjust if needed.*

---

## 📦 Pre‑trained Models

Download the weights from the release page:  
https://github.com/iheb-brini/SegClarity/releases/tag/model_weights

Place them under:

```
models/
├── cityscapes/
├── splitAB1/
└── UTP/
```

---

## 🗄 Datasets

### 📄 Document Datasets (UTP & splitAB1)
Already included in:
```
datasets/
```

### 🏙 Cityscapes Dataset (optional)
Download from: https://www.cityscapes-dataset.com/file-handling/?packageID=3

Required files:
- `leftImg8bit_trainvaltest.zip`
- `gtFine_trainvaltest.zip`

Extract into:

```
datasets/cityscapes/
├── leftImg8bit/
└── gtFine/
```

---

## 🧪 Running Experiments

Install Jupyter:
```bash
pip install jupyter notebook
```

Run:
```bash
jupyter notebook
```

### Provided notebooks:
- `01_Model_predictions_on_documents.ipynb` — Document segmentation evaluation  
- `02_Model_predictions_on_cityscapes.ipynb` — Urban scene segmentation  
- `03_Attributions_on_documents.ipynb` — Document explainability  
- `04_Attributions_on_cityscapes.ipynb` — Scene explainability  

Each notebook allows configuration of:
- Dataset  
- Architecture (UNet / LUNet)  
- Pretrained model choice  
- CPU/GPU runtime  

---

## ⚙️ System Requirements

- **Python** 3.10–3.12  
- **CUDA 12.6** (optional)  
- **RAM**: 8GB minimum, 16GB recommended  
- **Disk**: ~15GB for datasets + models  

---

## ❗ Troubleshooting

### 1. CUDA Out of Memory
- Reduce batch size  
- Use CPU mode  

### 2. Missing Model Weights
Ensure the structure is:
```
models/<dataset>/<architecture>/<model>.pth
```

### 3. Dataset Not Found
Check path:
```
datasets/<dataset-name>/
```

### 4. Import Errors
Reinstall dependencies:
```bash
pip install -r requirements.txt
```

---

## 📚 Citation

If you use **SegClarity** in your research, please cite:

```bibtex
@article{Brini2025SegClarity,
  author    = {Iheb Brini and others},
  title     = {SegClarity: Semantic Segmentation with Explainable AI},
  journal   = {Journal of Imaging},
  volume    = {11},
  number    = {12},
  pages     = {424},
  year      = {2025},
  publisher = {MDPI},
  doi       = {10.3390/jimaging11120424},
  url       = {https://www.mdpi.com/2313-433X/11/12/424}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.  
See the `LICENSE` file for more information.
