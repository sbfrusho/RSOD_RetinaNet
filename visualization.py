# visualization.py
"""
Visualization utilities for dataset analysis and results
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from collections import Counter

def visualize_augmentation_comparison(dataset, num_samples=5):
    """
    Compare original vs augmented images
    
    Args:
        dataset: Dataset object
        num_samples: Number of samples to visualize
    """
    indices = random.sample(range(len(dataset)), num_samples)

    for idx in indices:
        # Original image
        img_path = dataset.img_files[idx]
        original_image = np.array(Image.open(img_path).convert("RGB"))

        # Augmented image
        image, target = dataset[idx]
        aug_image = image.permute(1, 2, 0).numpy()
        boxes = target['boxes'].numpy()

        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Left: Original
        axes[0].imshow(original_image)
        axes[0].axis('off')
        axes[0].set_title(f"Original Image {idx}")

        # Right: Augmented (with bounding boxes)
        axes[1].imshow(aug_image)
        axes[1].set_title(f"Augmented Image {idx}")
        axes[1].axis('off')

        for box in boxes:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   linewidth=2, edgecolor='lime', facecolor='none')
            axes[1].add_patch(rect)

        plt.show()

def plot_bbox_heatmap(dataset):
    """
    Plot bounding box heatmap showing object density
    
    Args:
        dataset: Dataset object
    """
    heatmap = np.zeros((640, 640))

    for _, target in dataset:
        boxes = target['boxes'].numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(640, x2), min(640, y2)
            if x2 > x1 and y2 > y1:
                heatmap[y1:y2, x1:x2] += 1

    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("Bounding Box Heatmap (Object Density)")
    plt.axis('off')
    plt.colorbar(label='Object Count')
    plt.show()

def plot_class_distribution(dataset):
    """
    Plot class distribution in the dataset
    
    Args:
        dataset: Dataset object
    """
    label_counts = Counter()
    for _, target in dataset:
        labels = target['labels'].numpy()
        label_counts.update(labels)
    
    classes = ["aircraft", "oiltank", "overpass", "playground"]
    counts = [label_counts.get(i+1, 0) for i in range(len(classes))]
    
    # Print numerical counts
    print("="*40)
    print("Exact Class Counts:")
    for cls, count in zip(classes, counts):
        print(f"{cls:<10}: {count:>5} objects")
    print("="*40 + "\n")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=12)
    
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Objects", fontsize=12)
    plt.title("Class Distribution in RSOD Dataset", fontsize=14, pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_bbox_sizes(dataset):
    """
    Plot distribution of bounding box sizes
    
    Args:
        dataset: Dataset object
    """
    bbox_areas = []

    for _, target in dataset:
        boxes = target['boxes'].numpy()
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            bbox_areas.append(area)

    plt.figure(figsize=(8, 5))
    plt.hist(bbox_areas, bins=50, color='salmon', log=True)
    plt.xlabel("Bounding Box Area (pixels)")
    plt.ylabel("Frequency (log scale)")
    plt.title("Distribution of Bounding Box Sizes")
    plt.show()

def show_sample_image(dataset, index):
    """
    Show a sample image with bounding boxes
    
    Args:
        dataset: Dataset object
        index: Image index to display
    """
    image, target = dataset[index]
    image = image.permute(1, 2, 0).numpy()
    boxes = target['boxes'].numpy()

    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)

    plt.title(f"Sample Image {index} with Bounding Boxes")
    plt.show()

def plot_training_curves(train_losses, val_losses, epoch_mAPs=None):
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        epoch_mAPs: List of mAP values per epoch (optional)
    """
    num_epochs = len(train_losses)
    
    if epoch_mAPs:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        ax1.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
        ax1.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training and Validation Loss Curves")
        ax1.legend()
        ax1.grid(True)
        
        # mAP curve
        ax2.plot(range(1, num_epochs+1), epoch_mAPs, marker='o', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP@0.5')
        ax2.set_title('Validation mAP@0.5 over Epochs')
        ax2.grid(True)
        
        plt.tight_layout()
    else:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, num_epochs+1), train_losses, label="Train Loss")
        plt.plot(range(1, num_epochs+1), val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Curves")
        plt.legend()
        plt.grid(True)
    
    plt.show()

def plot_performance_bar_chart(class_aps, final_map):
    """
    Plot performance bar chart for each class
    
    Args:
        class_aps: Dictionary of class APs
        final_map: Final mAP value
    """
    classes = list(class_aps.keys())
    ap_values = list(class_aps.values())
    all_categories = classes + ["Final mAP"]
    all_values = ap_values + [final_map]

    # Color scheme
    colors = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(all_categories, all_values, color=colors)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height-0.03,
                f'{height:.3f}',
                ha='center', va='bottom',
                color='white', fontsize=12, fontweight='bold')

    plt.ylim(0.85, 1.0)
    plt.ylabel('Average Precision (AP@0.5)', fontsize=12)
    plt.title('RetinaNet Performance on RSOD Dataset', fontsize=14, pad=20)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Special formatting for final mAP
    bars[-1].set_edgecolor('black')
    bars[-1].set_linewidth(2)

    plt.tight_layout()
    plt.show()