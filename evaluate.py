# evaluate.py
"""
Evaluation script for RetinaNet using COCO metrics
"""

import os
import json
import torch
from torch.utils.data import DataLoader
import contextlib
import io
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import RSODDataset, collate_fn
from transforms import get_val_transform
from model import get_model, load_model

def evaluate_model(model, val_loader, dataset, device, output_file="predictions.json"):
    """
    Evaluate model and generate predictions
    
    Args:
        model: Trained RetinaNet model
        val_loader: Validation data loader
        dataset: Validation dataset
        device: Device to run on
        output_file: Output file for predictions
    
    Returns:
        List of predictions in COCO format
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for idx, (imgs, targets) in enumerate(val_loader):
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            
            for j, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                
                # Filter low confidence predictions
                keep = scores > 0.2
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                
                # Convert to COCO format
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        "image_id": idx * len(imgs) + j,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })
    
    # Save predictions
    with open(output_file, "w") as f:
        json.dump(results, f)
    
    print(f"Saved {len(results)} predictions to {output_file}")
    return results

def create_coco_gt(dataset):
    """
    Create COCO ground truth format
    
    Args:
        dataset: Validation dataset
    
    Returns:
        COCO ground truth object
    """
    coco_gt = COCO()
    
    # Create dataset structure
    gt_dataset = {
        "info": {}, 
        "licenses": [],
        "images": [{"id": i} for i in range(len(dataset))],
        "categories": [
            {"id": 1, "name": "aircraft"},
            {"id": 2, "name": "oiltank"},
            {"id": 3, "name": "overpass"},
            {"id": 4, "name": "playground"}
        ],
        "annotations": []
    }

    # Add annotations
    ann_id = 0
    for img_id in range(len(dataset)):
        _, target = dataset[img_id]
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()

        for k in range(len(boxes)):
            x1, y1, x2, y2 = boxes[k]
            gt_dataset['annotations'].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(labels[k]),
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            ann_id += 1

    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()
    return coco_gt

def compute_coco_metrics(coco_gt, predictions_file):
    """
    Compute COCO evaluation metrics
    
    Args:
        coco_gt: COCO ground truth object
        predictions_file: Path to predictions JSON file
    
    Returns:
        Dictionary of metrics
    """
    # Load predictions
    coco_dt = coco_gt.loadRes(predictions_file)
    
    # Evaluate
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Suppress stdout for cleaner output
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        "mAP_50_95": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "mAP_75": coco_eval.stats[2],
        "mAP_small": coco_eval.stats[3],
        "mAP_medium": coco_eval.stats[4],
        "mAP_large": coco_eval.stats[5],
        "AR_1": coco_eval.stats[6],
        "AR_10": coco_eval.stats[7],
        "AR_100": coco_eval.stats[8],
        "AR_small": coco_eval.stats[9],
        "AR_medium": coco_eval.stats[10],
        "AR_large": coco_eval.stats[11]
    }
    
    return metrics, coco_eval

def compute_per_class_ap(coco_gt, coco_eval, dataset):
    """
    Compute per-class AP
    
    Args:
        coco_gt: COCO ground truth object
        coco_eval: COCO evaluation object
        dataset: Dataset object
    
    Returns:
        Dictionary of per-class AP values
    """
    class_aps = {}
    class_names = ["aircraft", "oiltank", "overpass", "playground"]
    
    for i, name in enumerate(class_names, 1):
        coco_eval.params.catIds = [i]
        coco_eval.evaluate()
        coco_eval.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.summarize()
        class_aps[name] = coco_eval.stats[1]  # AP@0.5
    
    return class_aps

def plot_confusion_matrix(model, val_loader, device, class_names):
    """
    Plot confusion matrix
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        device: Device
        class_names: List of class names
    """
    y_true, y_pred = [], []
    
    model.eval()
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)
            
            for i, output in enumerate(outputs):
                pred_labels = output['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                # Filter predictions by confidence
                high_conf_preds = pred_labels[pred_scores > 0.5]
                
                y_true.extend(gt_labels)
                # Match predictions to ground truth (simplified)
                if len(high_conf_preds) > 0:
                    y_pred.extend(high_conf_preds[:len(gt_labels)])
                else:
                    y_pred.extend([0] * len(gt_labels))  # Background class
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4])
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def main():
    # Configuration
    val_img_dir = "/path/to/val/images"
    val_ann_dir = "/path/to/val/annotations"
    model_path = "retinanet_rsod.pth"
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and dataloader
    val_dataset = RSODDataset(val_img_dir, val_ann_dir, get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, 
                           num_workers=2, collate_fn=collate_fn)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load model
    model = get_model(num_classes=5)
    model = load_model(model, model_path, device)
    model.to(device)
    
    print("Starting evaluation...")
    
    # Generate predictions
    predictions = evaluate_model(model, val_loader, val_dataset, device)
    
    # Create COCO ground truth
    coco_gt = create_coco_gt(val_dataset)
    
    # Compute COCO metrics
    metrics, coco_eval = compute_coco_metrics(coco_gt, "predictions.json")
    
    # Print overall metrics
    print("\n" + "="*50)
    print("COCO Evaluation Results")
    print("="*50)
    print(f"mAP@[IoU=0.50:0.95]: {metrics['mAP_50_95']:.4f}")
    print(f"mAP@0.5:             {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75:            {metrics['mAP_75']:.4f}")
    print(f"AR@100:              {metrics['AR_100']:.4f}")
    
    # Compute per-class AP
    class_aps = compute_per_class_ap(coco_gt, coco_eval, val_dataset)
    
    print("\nPer-Class AP@0.5:")
    print("-" * 30)
    for cls, ap in class_aps.items():
        print(f"{cls:<12}: {ap:.4f}")
    
    # Plot confusion matrix
    class_names = ["aircraft", "oiltank", "overpass", "playground"]
    plot_confusion_matrix(model, val_loader, device, class_names)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()