# RetinaNet for RSOD Dataset

A complete implementation of RetinaNet for object detection on the Remote Sensing Object Detection (RSOD) dataset with CLAHE enhancement and comprehensive evaluation tools.

## Features

- **CLAHE Enhancement**: Contrast Limited Adaptive Histogram Equalization for improved image contrast
- **Data Augmentation**: Random brightness/contrast, horizontal flip, rotation
- **COCO Evaluation**: Complete evaluation using COCO metrics
- **Visualization Tools**: Dataset analysis, training curves, confusion matrices
- **Batch & Single Inference**: Support for both batch processing and single image inference
- **Modular Design**: Clean, organized code structure

## Dataset Classes

- Aircraft
- Oil Tank
- Overpass
- Playground

## Project Structure

```
├── main.py                 # Main script with interactive menu
├── config.py              # Configuration settings
├── data_preprocessing.py   # YOLO to Pascal VOC conversion
├── dataset.py             # Dataset class for loading data
├── transforms.py          # Image transformations and CLAHE
├── model.py               # RetinaNet model definition
├── train.py               # Training script
├── evaluate.py            # Evaluation with COCO metrics
├── batch_inference.py     # Batch inference script
├── single_inference.py    # Single image inference
├── visualization.py       # Visualization utilities
└── requirements.txt       # Package dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd retinanet-rsod
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode
Run the main script for an interactive menu:
```bash
python main.py
```

### Direct Commands

**Training:**
```bash
python main.py --mode train
```

**Evaluation:**
```bash
python main.py --mode evaluate
```

**Single Image Inference:**
```bash
python single_inference.py --image /path/to/image.jpg --weights retinanet_rsod.pth
```

**Batch Inference:**
```bash
python batch_inference.py --dataset_dir /path/to/images --ann_dir /path/to/annotations --weights retinanet_rsod.pth
```

## Configuration

Edit `config.py` to set your dataset paths and training parameters:

```python
class Config:
    # Dataset paths
    TRAIN_IMG_DIR = "/path/to/train/images"
    TRAIN_ANN_DIR = "/path/to/train/annotations"
    VAL_IMG_DIR = "/path/to/val/images"
    VAL_ANN_DIR = "/path/to/val/annotations"
    
    # Training parameters
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.005
    # ... more parameters
```

## Data Preprocessing

If you have YOLO format annotations, convert them to Pascal VOC format:

```python
from data_preprocessing import yolo_to_pascal_voc

yolo_to_pascal_voc(
    image_dir="/path/to/images",
    label_dir="/path/to/yolo/labels", 
    output_dir="/path/to/pascal/annotations",
    classes=['aircraft', 'oiltank', 'overpass', 'playground']
)
```

## Training

The training script includes:
- CLAHE image enhancement
- Data augmentation (brightness/contrast, flips, rotations)
- Cosine annealing learning rate schedule
- Automatic loss tracking and visualization

**Training Features:**
- Automatic mixed precision training (if available)
- Progress tracking with loss curves
- Model checkpointing
- Validation during training

## Evaluation

Comprehensive evaluation using COCO metrics:
- mAP@0.5:0.95
- mAP@0.5
- Per-class Average Precision
- Confusion matrices
- Detection visualizations

## Model Performance
| Metric              | Value  |
|---------------------|--------|
| mAP@[IoU=0.50:0.95] | **0.7466** |
| mAP@0.5 (FINAL)     | **0.9633** |
| AR@100 (all)        | **0.7923** |
| AR@100 (large)      | **0.8063** |


Based on the thesis results:
- **Aircraft**: 94.17% AP@0.5
- **Oil Tank**: 97.94% AP@0.5  
- **Overpass**: 95.71% AP@0.5
- **Playground**: 93.63% AP@0.5
- **Overall mAP@0.5**: 93.63%

## Key Components

### CLAHE Enhancement
```python
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        # Enhances local contrast while preventing over-amplification
```

### Data Augmentation
- Random brightness/contrast (±20%)
- Horizontal flips (50% probability)
- Random 90° rotations (50% probability)
- Resize to 640x640
- ImageNet normalization

### Model Architecture
- **Backbone**: ResNet-50 with Feature Pyramid Network (FPN)
- **Head**: RetinaNet classification and regression heads
- **Anchors**: Multi-scale anchor boxes
- **Loss**: Focal loss for classification, smooth L1 for regression

## Visualization Tools

The project includes comprehensive visualization utilities:

```python
# Dataset analysis
plot_class_distribution(dataset)
plot_bbox_heatmap(dataset)
plot_bbox_sizes(dataset)

# Training monitoring
plot_training_curves(train_losses, val_losses)

# Results visualization  
show_sample_image(dataset, index)
visualize_augmentation_comparison(dataset)
```

## Usage Examples

### Training a Model
```python
from train import main as train_model
train_model()
```

### Single Prediction
```python
from single_inference import main as single_inference
# Run with: python single_inference.py --image image.jpg --weights model.pth
```

### Batch Processing
```python
from batch_inference import main as batch_inference  
# Run with: python batch_inference.py --dataset_dir images/ --weights model.pth
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV 4.5+
- albumentations 1.0+
- pycocotools 2.0+
- See `requirements.txt` for complete list

## Hardware Requirements

**Minimum:**
- 8GB RAM
- 4GB GPU memory (GTX 1060 or equivalent)

**Recommended:**
- 16GB+ RAM  
- 8GB+ GPU memory (RTX 2070 or better)
- SSD storage for faster data loading

## Tips for Best Results

1. **Data Quality**: Ensure annotations are accurate and consistent
2. **Augmentation**: Adjust augmentation parameters based on your dataset characteristics
3. **Learning Rate**: Start with 0.005 and adjust based on loss curves
4. **Batch Size**: Use largest batch size that fits in GPU memory
5. **Epochs**: Monitor validation loss to avoid overfitting

## Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory**: Reduce batch size in `config.py`
2. **Slow Training**: Increase `num_workers` in DataLoader
3. **Poor Performance**: Check data augmentation parameters
4. **Import Errors**: Ensure all requirements are installed

## Contributing

Feel free to open issues or submit pull requests for improvements!

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:
```
@misc{retinanet-rsod,
  title={RetinaNet Implementation for RSOD Dataset},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/your-repo}}
}
```
