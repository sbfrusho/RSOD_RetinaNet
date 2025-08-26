# batch_inference.py
"""
Batch inference script for processing multiple images
"""

import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import warnings
import gc
import contextlib
import io

from rich.console import Console
from rich.table import Table
from rich.progress import track

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset import RSODDataset, collate_fn
from transforms import get_val_transform
from model import get_model, load_model

# Setup
console = Console()
warnings.filterwarnings("ignore", category=UserWarning)

def soft_nms_pytorch(boxes, scores, iou_threshold=0.5):
    """
    Placeholder for Soft-NMS (using regular NMS for now)
    """
    from torchvision.ops import nms
    return nms(boxes, scores, iou_threshold)

def batch_inference(model, val_loader, dataset, device, confidence_threshold=0.2):
    """
    Perform batch inference on validation set
    
    Args:
        model: Trained RetinaNet model
        val_loader: Validation data loader
        dataset: Dataset object
        device: Device to run on
        confidence_threshold: Minimum confidence threshold
    
    Returns:
        results: List of predictions
        readable_results: List of human-readable predictions
    """
    model.eval()
    results = []
    readable_results = []
    
    # Open file for readable predictions
    with open("predictions.txt", "w") as txt_file:
        txt_file.write("image_name,image_id,true_labels,pred_label,score\n")
        
        with torch.no_grad():
            for idx, (imgs, targets) in enumerate(track(val_loader, description="🚀 Running inference...")):
                imgs = [img.to(device) for img in imgs]
                outputs = model(imgs)
                
                for j, output in enumerate(outputs):
                    boxes = output['boxes'].cpu()
                    scores = output['scores'].cpu()
                    labels = output['labels'].cpu()
                    
                    # Apply Soft-NMS (or regular NMS)
                    keep = soft_nms_pytorch(boxes, scores)
                    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
                    
                    # Get image info
                    image_idx = idx * len(imgs) + j
                    img_name = os.path.basename(dataset.img_files[image_idx])
                    
                    # Get ground truth labels
                    true_labels = list(set(targets[j]['labels'].cpu().numpy()))
                    true_str = '/'.join([dataset.id_to_name[l] for l in true_labels]) if true_labels else "NA"
                    
                    # Process predictions
                    for b, s, l in zip(boxes, scores, labels):
                        if s < confidence_threshold:
                            continue
                            
                        pred_name = dataset.id_to_name[int(l)]
                        
                        # Save readable format
                        txt_file.write(f"{img_name},{image_idx},{true_str},{pred_name},{s:.3f}\n")
                        readable_results.append({
                            "image_name": img_name,
                            "image_id": image_idx,
                            "true_labels": true_str,
                            "predicted_class": pred_name,
                            "confidence": float(s)
                        })
                        
                        # Save COCO format
                        results.append({
                            "image_id": image_idx,
                            "category_id": int(l),
                            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
                            "score": float(s)
                        })
    
    return results, readable_results

def evaluate_with_coco_metrics(results, dataset):
    """
    Evaluate using COCO metrics
    
    Args:
        results: Predictions in COCO format
        dataset: Dataset object
    
    Returns:
        Dictionary of metrics
    """
    # Save predictions
    with open("retina_predictions.json", "w") as f:
        json.dump(results, f)
    
    # Create COCO ground truth
    coco_gt = COCO()
    coco_gt.dataset = {
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
    
    # Add ground truth annotations
    ann_id = 0
    for i in range(len(dataset)):
        _, target = dataset[i]
        for b, l in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = b.numpy()
            coco_gt.dataset['annotations'].append({
                "id": ann_id,
                "image_id": i,
                "category_id": int(l),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": float((x2 - x1) * (y2 - y1)),
                "iscrowd": 0
            })
            ann_id += 1
    
    coco_gt.createIndex()
    
    # Load predictions and evaluate
    coco_dt = coco_gt.loadRes("retina_predictions.json")
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Suppress stdout for cleaner output
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    
    # Extract key metrics
    metrics = {
        "mAP_50_95": coco_eval.stats[0],
        "mAP_50": coco_eval.stats[1],
        "AR_100": coco_eval.stats[8],
        "AR_100_large": coco_eval.stats[10]
    }
    
    # Per-class AP
    class_aps = {}
    for i in range(1, 5):
        coco_eval.params.catIds = [i]
        coco_eval.evaluate()
        coco_eval.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_eval.summarize()
        class_name = dataset.id_to_name[i]
        class_aps[class_name] = coco_eval.stats[1]
    
    return metrics, class_aps

def print_results(metrics, class_aps, num_predictions):
    """
    Print evaluation results in a nice format
    """
    console.rule("[bold cyan]Batch Inference Results")
    
    # Overall metrics
    summary = Table(title="📊 Overall COCO Metrics")
    summary.add_column("Metric", justify="left")
    summary.add_column("Value", justify="center")
    summary.add_row("mAP@[IoU=0.50:0.95]", f"{metrics['mAP_50_95']:.4f}")
    summary.add_row("mAP@0.5 (FINAL)", f"{metrics['mAP_50']:.4f}")
    summary.add_row("AR@100 (all)", f"{metrics['AR_100']:.4f}")
    summary.add_row("AR@100 (large)", f"{metrics['AR_100_large']:.4f}")
    console.print(summary)
    
    # Per-class results
    class_table = Table(title="📊 Per-Class AP@0.5")
    class_table.add_column("Class", justify="left")
    class_table.add_column("AP@0.5", justify="center")
    
    for cls, ap in class_aps.items():
        class_table.add_row(cls, f"{ap:.4f}")
    console.print(class_table)
    
    console.print(f"\n✅ [green]Generated {num_predictions} predictions")
    console.print(f"✅ [green]Saved readable predictions to predictions.txt")
    console.print(f"✅ [green]Saved COCO format predictions to retina_predictions.json")

def main():
    parser = argparse.ArgumentParser(description="Batch inference for RetinaNet")
    parser.add_argument("--dataset_dir", required=True, help="Path to validation images directory")
    parser.add_argument("--ann_dir", required=True, help="Path to annotations directory")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference")
    parser.add_argument("--confidence", type=float, default=0.2, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [bold green]{device}")
    
    # Dataset
    dataset = RSODDataset(args.dataset_dir, args.ann_dir, transforms=get_val_transform())
    val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                           collate_fn=collate_fn)
    
    console.print(f"Dataset loaded: [bold blue]{len(dataset)} images")
    
    # Model
    model = get_model(num_classes=5)
    model = load_model(model, args.weights, device)
    model.to(device)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Run inference
    console.print("\nStarting batch inference...")
    results, readable_results = batch_inference(model, val_loader, dataset, device, args.confidence)
    
    # Evaluate
    console.print("Computing COCO metrics...")
    metrics, class_aps = evaluate_with_coco_metrics(results, dataset)
    
    # Print results
    print_results(metrics, class_aps, len(results))

if __name__ == "__main__":
    main()