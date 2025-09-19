# Medical Image Segmentation Project

## Overview
This project implements medical image segmentation using deep learning techniques, specifically focusing on brain tumor segmentation from MRI scans. The project serves as a practical learning experience for biomedical image computing.

## Features
- **Data Preprocessing**: Automated preprocessing pipeline for medical images
- **Model Architecture**: U-Net implementation with various configurations
- **Training Pipeline**: Complete training workflow with validation
- **Evaluation Metrics**: Dice coefficient, IoU, and other medical imaging metrics
- **Visualization Tools**: Interactive visualization of segmentation results
- **Web Interface**: Simple web-based interface for model inference

## Project Structure
```
medical-image-segmentation/
├── data/                   # Dataset storage
├── models/                 # Model definitions
├── utils/                  # Utility functions
├── training/               # Training scripts
├── evaluation/             # Evaluation and metrics
├── visualization/          # Visualization tools
├── web_app/               # Web interface
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Python dependencies
└── config/               # Configuration files
```

## Setup Instructions

### Necessary Prerequisites
- Python 3.8+
- 8GB+ RAM
### Recommended Prerequisites
- CUDA-compatible GPU

### Installation
1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Setup
1. Download the BraTS dataset (requires registration)
2. Place data in the `data/` directory
3. Run preprocessing script:
   ```bash
   python utils/preprocess.py
   ```

## Usage

### Training
```bash
python training/train.py --config config/training_config.yaml
```

### Evaluation
```bash
python evaluation/evaluate.py --model_path models/best_model.pth
```

### Web Interface
```bash
python web_app/app.py
```

## Model Architecture
- **U-Net**: Primary architecture with skip connections
- **Attention Mechanisms**: Optional attention gates for better focus
- **Data Augmentation**: Rotation, scaling, intensity variations
- **Loss Functions**: Dice Loss, Focal Loss, and combinations

## Evaluation Metrics
- Dice Coefficient (F1-Score)
- Intersection over Union (IoU)
- Hausdorff Distance
- Sensitivity and Specificity

## Contributing
This is a learning project. Feel free to experiment with different architectures, datasets, and techniques.

## License
MIT License - for educational purposes only. 