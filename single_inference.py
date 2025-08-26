# single_inference.py
"""
Single image inference script for RetinaNet
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import argparse

from model import get_model, load_model
from transforms import get_inference_transform

def load_image(image_path):
    """
    Load and preprocess image for inference
    
    Args:
        image_path: Path to input image
    
    Returns:
        original_image: PIL Image (original)
        processed_image: Tensor (preprocessed)
    """
    original_image = Image.open(image_path).convert("RGB")
    
    # Apply transforms
    transform = get_inference_transform()
    processed = transform(image=np.array(original_image))
    processed_image = processed['image'].unsqueeze(0)  # Add batch dimension
    
    return original_image, processed_image

def draw_predictions(image, predictions, class_names, confidence_threshold=0.5):
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: PIL Image
        predictions: Model predictions
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
    
    Returns:
        PIL Image with drawn predictions
    """
    # Resize image to match model input size
    img_with_boxes = image.copy().resize((640, 640))
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to use a larger font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Colors for each class
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    detection_count = 0
    
    for box, score, label in zip(boxes, scores, labels):
        if score < confidence_threshold:
            continue
            
        x1, y1, x2, y2 = box
        cls_name = class_names[label]
        color = colors[label % len(colors)]
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label with background
        text = f"{cls_name} {score:.2f}"
        bbox = draw.textbbox((x1, y1-25), text, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1-25), text, fill="white", font=font)
        
        detection_count += 1
    
    return img_with_boxes, detection_count

def print_detection_results(predictions, class_names, confidence_threshold=0.5):
    """
    Print detection results to console
    
    Args:
        predictions: Model predictions
        class_names: List of class names
        confidence_threshold: Minimum confidence to display
    """
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    
    detection_count = 0
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < confidence_threshold:
            continue
            
        x1, y1, x2, y2 = box
        cls_name = class_names[label]
        detection_count += 1
        
        print(f"\nDetection {detection_count}:")
        print(f"  Class: {cls_name}")
        print(f"  Confidence: {score:.4f}")
        print(f"  Bounding Box: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")
        print(f"  Box Size: {x2-x1:.1f} x {y2-y1:.1f} pixels")
    
    if detection_count == 0:
        print(f"\nNo detections above confidence threshold of {confidence_threshold}")
    else:
        print(f"\nTotal detections: {detection_count}")
    
    print("="*60)

def visualize_results(original_image, result_image, save_path=None):
    """
    Display original and result images side by side
    
    Args:
        original_image: Original PIL image
        result_image: PIL image with detections
        save_path: Optional path to save the result
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original image
    axes[0].imshow(original_image.resize((640, 640)))
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis("off")
    
    # Result image
    axes[1].imshow(result_image)
    axes[1].set_title("Detections", fontsize=14)
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Result saved to: {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Single image inference for RetinaNet")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--weights", required=True, help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--save", help="Path to save result image")
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Class names (including background at index 0)
    class_names = ["__background__", "aircraft", "oiltank", "overpass", "playground"]
    
    print("Loading model...")
    # Load model
    model = get_model(num_classes=5)
    model = load_model(model, args.weights, device)
    model.to(device)
    model.eval()
    
    print(f"Loading image: {args.image}")
    # Load and preprocess image
    import numpy as np
    original_image, processed_image = load_image(args.image)
    processed_image = processed_image.to(device)
    
    print("Running inference...")
    # Run inference
    with torch.no_grad():
        predictions = model(processed_image)[0]
    
    # Print results
    print_detection_results(predictions, class_names, args.confidence)
    
    # Draw predictions
    result_image, detection_count = draw_predictions(
        original_image, predictions, class_names, args.confidence
    )
    
    # Visualize results
    save_path = args.save if args.save else None
    visualize_results(original_image, result_image, save_path)
    
    print(f"\nInference completed! Found {detection_count} objects.")

if __name__ == "__main__":
    main()