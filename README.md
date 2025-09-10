# SegClarity

SegClarity is a comprehensive framework for semantic segmentation with explainable AI capabilities, supporting both document segmentation and urban scene understanding tasks.

## Overview

This project provides:
- **Document Segmentation**: Models trained on UTP and splitAB1 datasets for document layout analysis
- **Urban Scene Segmentation**: Models trained on Cityscapes dataset for street scene understanding
- **Explainable AI**: Attribution methods for understanding model decisions
- **Visualization Tools**: Comprehensive visualization of predictions and attributions

## Project Structure

```
SegClarity/
├── Modules/                    # Core framework modules
│   ├── Architecture/          # Model architectures (UNet, LUNet)
│   ├── Dataset/              # Dataset handling utilities
│   ├── CityscapeDataset/     # Cityscapes-specific dataset tools
│   ├── ModelXAI/            # Explainable AI methods
│   ├── Attribution/         # Attribution computation
│   ├── Visualization/       # Visualization utilities
│   └── ...
├── Notebooks/                # Jupyter notebooks for experiments
│   ├── 01_Model_predictions_on_documents.ipynb
│   ├── 02_Model_predictions_on_cityscapes.ipynb
│   ├── 03_Attributions_on_documents.ipynb
│   └── 04_Attributions_on_cityscapes.ipynb
├── models/                   # Pre-trained model weights
├── datasets/                 # Dataset storage
└── requirements.txt          # Python dependencies
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/iheb-brini/SegClarity.git
cd SegClarity
```

### 2. Install Dependencies

Create a virtual environment and install the required packages:

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The requirements include PyTorch with CUDA 12.6 support. If you don't have CUDA or need a different version, modify the PyTorch installation in `requirements.txt`.

**Key Dependencies**:
- **PyTorch & TorchVision**: Deep learning framework with CUDA support
- **Captum**: Model interpretability and attribution methods
- **Albumentations**: Advanced image augmentation library (used for Cityscapes)
- **Scikit-image**: Image processing utilities (used for Otsu thresholding, resizing)
- **OpenCV**: Computer vision operations
- **Matplotlib**: Visualization and plotting
- **PIL/Pillow**: Image loading and processing
- **Pytest**: Testing framework (for evaluation modules)

### 3. Download Pre-trained Models

Download the pre-trained model weights from the releases:

```bash
# Create models directory if it doesn't exist
mkdir -p models

# Download model weights from GitHub releases
# Visit: https://github.com/iheb-brini/SegClarity/releases/tag/model_weights
# Download the model weights archive and extract to the models/ folder
```

**Expected model structure after download:**
```
models/
├── cityscapes/
│   └── unet/
│       └── best_model.pth
├── splitAB1/
│   ├── lunet/
│   │   ├── finetuned_models_minloss/
│   │   └── from_scratch_models/
│   └── unet/
│       ├── finetuned_models_minloss/
│       └── from_scratch_models/
└── UTP/
    ├── lunet/
    │   └── from_scratch_models/
    └── unet/
        └── from_scratch_models/
```

### 4. Download Datasets

#### Document Datasets (UTP and splitAB1)
The document datasets (UTP and splitAB1) are already included in the repository under the `datasets/` folder.

#### Cityscapes Dataset
Download the Cityscapes dataset for urban scene segmentation:

1. **Register and Login**: Visit [Cityscapes Dataset](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
2. **Download**: Download the following packages:
   - `leftImg8bit_trainvaltest.zip` (11GB) - Training, validation, and test images
   - `gtFine_trainvaltest.zip` (241MB) - Fine annotations
3. **Extract**: Extract the downloaded files to `datasets/cityscapes/`

**Expected Cityscapes structure:**
```
datasets/cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

## Running Experiments

### Jupyter Notebooks
Install `Jupyter notebook` (if missing):

```bash
pip install jupyter notebook 
```
Start Jupyter and run the experiment notebooks:

```bash
jupyter notebook
```

#### Available Notebooks:

1. **`01_Model_predictions_on_documents.ipynb`**
   - Evaluates document segmentation models (LUNet, UNet)
   - Works with UTP and splitAB1 datasets
   - Visualizes predictions vs ground truth

2. **`02_Model_predictions_on_cityscapes.ipynb`**
   - Evaluates urban scene segmentation models
   - Works with Cityscapes dataset
   - Provides semantic segmentation results

3. **`03_Attributions_on_documents.ipynb`**
   - Computes and visualizes attributions for document models
   - Uses various XAI methods (GradCAM, Integrated Gradients, etc.)
   - Analyzes model decision-making on document layouts

4. **`04_Attributions_on_cityscapes.ipynb`**
   - Computes and visualizes attributions for urban scene models
   - Explains model predictions on street scenes
   - Provides insights into what the model focuses on

### Notebook Configuration

Each notebook allows you to configure:
- **Dataset type**: Choose between available datasets
- **Model architecture**: Select UNet or LUNet
- **Model variant**: Choose from-scratch or fine-tuned models
- **Device**: CPU or GPU (if available)

## System Requirements

- **Python**: >=3.10, <3.13
- **CUDA**: 12.6 (optional, for GPU acceleration)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 15GB for datasets and models

## Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Missing Models**: Ensure model weights are downloaded and placed in correct directories
3. **Dataset Not Found**: Verify dataset paths and structure
4. **Import Errors**: Check that all dependencies are installed correctly

### Getting Help:

- Check the notebook documentation for specific usage instructions
- Verify file paths and directory structures match the expected layout
- Ensure all dependencies are properly installed

<!-- 
## Citation

If you use this work in your research, please cite:

```bibtex
@software{segclarity,
  title={SegClarity: Semantic Segmentation with Explainable AI},
  author={Iheb Brini},
  year={2025},
  url={https://github.com/iheb-brini/SegClarity}
}
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
-->