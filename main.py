# main.py
"""
Main script to run different modes of the RetinaNet system
"""

import argparse
import sys
import os

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config

def run_preprocessing():
    """Run data preprocessing to convert YOLO to Pascal VOC format"""
    from data_preprocessing import yolo_to_pascal_voc
    
    print("Starting data preprocessing...")
    
    # You'll need to modify these paths according to your YOLO dataset structure
    yolo_train_images = "/path/to/yolo/train/images"
    yolo_train_labels = "/path/to/yolo/train/labels"
    yolo_val_images = "/path/to/yolo/val/images"
    yolo_val_labels = "/path/to/yolo/val/labels"
    
    # Convert training data
    print("Converting training data...")
    yolo_to_pascal_voc(
        image_dir=yolo_train_images,
        label_dir=yolo_train_labels,
        output_dir=Config.TRAIN_ANN_DIR,
        classes=Config.CLASSES
    )
    
    # Convert validation data
    print("Converting validation data...")
    yolo_to_pascal_voc(
        image_dir=yolo_val_images,
        label_dir=yolo_val_labels,
        output_dir=Config.VAL_ANN_DIR,
        classes=Config.CLASSES
    )
    
    print("✅ Data preprocessing completed!")

def run_training():
    """Run model training"""
    from train import main as train_main
    
    print("Starting training...")
    Config.print_config()
    
    if Config.verify_paths():
        train_main()
    else:
        print("❌ Cannot start training due to missing paths. Please check config.py")

def run_evaluation():
    """Run model evaluation"""
    from evaluate import main as eval_main
    
    print("Starting evaluation...")
    
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model weights not found: {Config.MODEL_SAVE_PATH}")
        print("Please train the model first or provide the correct path.")
        return
    
    eval_main()

def run_batch_inference():
    """Run batch inference"""
    print("For batch inference, use:")
    print(f"python batch_inference.py --dataset_dir {Config.VAL_IMG_DIR} --ann_dir {Config.VAL_ANN_DIR} --weights {Config.MODEL_SAVE_PATH}")

def run_single_inference():
    """Run single image inference"""
    image_path = input("Enter the path to the image: ").strip()
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model weights not found: {Config.MODEL_SAVE_PATH}")
        print("Please train the model first or provide the correct path.")
        return
    
    print("For single image inference, use:")
    print(f"python single_inference.py --image {image_path} --weights {Config.MODEL_SAVE_PATH}")

def run_visualization():
    """Run dataset visualization"""
    from dataset import RSODDataset, collate_fn
    from transforms import get_train_transform, get_val_transform
    from visualization import (
        visualize_augmentation_comparison,
        plot_bbox_heatmap,
        plot_class_distribution,
        plot_bbox_sizes,
        show_sample_image
    )
    
    print("Starting dataset visualization...")
    
    if not Config.verify_paths():
        print("❌ Cannot run visualization due to missing paths.")
        return
    
    # Load datasets
    from torch.utils.data import DataLoader
    train_dataset = RSODDataset(Config.TRAIN_IMG_DIR, Config.TRAIN_ANN_DIR, get_train_transform())
    val_dataset = RSODDataset(Config.VAL_IMG_DIR, Config.VAL_ANN_DIR, get_val_transform())
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    while True:
        print("\nVisualization Options:")
        print("1. Augmentation comparison")
        print("2. Bounding box heatmap")
        print("3. Class distribution")
        print("4. Bounding box sizes")
        print("5. Sample image with boxes")
        print("6. Back to main menu")
        
        choice = input("Select option (1-6): ").strip()
        
        if choice == "1":
            visualize_augmentation_comparison(train_dataset, num_samples=3)
        elif choice == "2":
            plot_bbox_heatmap(train_dataset)
        elif choice == "3":
            plot_class_distribution(train_dataset)
        elif choice == "4":
            plot_bbox_sizes(train_dataset)
        elif choice == "5":
            idx = input(f"Enter image index (0-{len(train_dataset)-1}): ").strip()
            try:
                idx = int(idx)
                if 0 <= idx < len(train_dataset):
                    show_sample_image(train_dataset, idx)
                else:
                    print("Invalid index!")
            except ValueError:
                print("Please enter a valid number!")
        elif choice == "6":
            break
        else:
            print("Invalid choice!")

def main():
    """Main function with menu system"""
    parser = argparse.ArgumentParser(description="RetinaNet for RSOD Dataset")
    parser.add_argument("--mode", choices=[
        "preprocess", "train", "evaluate", "batch_inference", 
        "single_inference", "visualize", "interactive"
    ], default="interactive", help="Mode to run")
    
    args = parser.parse_args()
    
    if args.mode == "interactive":
        # Interactive mode
        while True:
            print("\n" + "="*50)
            print("RETINANET RSOD DETECTION SYSTEM")
            print("="*50)
            print("1. Data Preprocessing (YOLO to Pascal VOC)")
            print("2. Train Model")
            print("3. Evaluate Model")
            print("4. Batch Inference")
            print("5. Single Image Inference")
            print("6. Dataset Visualization")
            print("7. Show Configuration")
            print("8. Exit")
            print("="*50)
            
            choice = input("Select option (1-8): ").strip()
            
            if choice == "1":
                run_preprocessing()
            elif choice == "2":
                run_training()
            elif choice == "3":
                run_evaluation()
            elif choice == "4":
                run_batch_inference()
            elif choice == "5":
                run_single_inference()
            elif choice == "6":
                run_visualization()
            elif choice == "7":
                Config.print_config()
                Config.verify_paths()
            elif choice == "8":
                print("Goodbye!")
                break
            else:
                print("Invalid choice! Please select 1-8.")
    else:
        # Direct mode
        if args.mode == "preprocess":
            run_preprocessing()
        elif args.mode == "train":
            run_training()
        elif args.mode == "evaluate":
            run_evaluation()
        elif args.mode == "batch_inference":
            run_batch_inference()
        elif args.mode == "single_inference":
            run_single_inference()
        elif args.mode == "visualize":
            run_visualization()

if __name__ == "__main__":
    main()