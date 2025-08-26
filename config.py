# config.py
"""
Configuration file for RSOD dataset and RetinaNet training
"""

import os

class Config:
    """Configuration class for the project"""
    
    # Dataset paths (modify these according to your setup)
    TRAIN_IMG_DIR = "/home/saku/2010776109_RUSHO/RSOD_YOLO/train/images"
    TRAIN_ANN_DIR = "/home/saku/2010776109_RUSHO/RSOD_YOLO/train/Annotations"
    VAL_IMG_DIR = "/home/saku/2010776109_RUSHO/RSOD_YOLO/val/images"
    VAL_ANN_DIR = "/home/saku/2010776109_RUSHO/RSOD_YOLO/val/Annotations"
    
    # Class configuration
    CLASSES = ['aircraft', 'oiltank', 'overpass', 'playground']
    NUM_CLASSES = 5  # 4 classes + background
    CLASS_DICT = {"aircraft": 1, "oiltank": 2, "overpass": 3, "playground": 4}
    
    # Training parameters
    BATCH_SIZE = 4
    VAL_BATCH_SIZE = 2
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    
    # Model parameters
    INPUT_SIZE = 640
    SCORE_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.2
    
    # CLAHE parameters
    CLAHE_CLIP_LIMIT = 2.0
    CLAHE_TILE_GRID_SIZE = (8, 8)
    
    # Augmentation parameters
    BRIGHTNESS_CONTRAST_LIMIT = 0.2
    HORIZONTAL_FLIP_PROB = 0.5
    ROTATION_PROB = 0.5
    
    # ImageNet normalization
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Output paths
    MODEL_SAVE_PATH = "retinanet_rsod.pth"
    PREDICTIONS_FILE = "predictions.json"
    READABLE_PREDICTIONS_FILE = "predictions.txt"
    
    # Device configuration
    USE_CUDA = True
    NUM_WORKERS = 2
    
    @classmethod
    def verify_paths(cls):
        """Verify that all dataset paths exist"""
        paths_to_check = [
            cls.TRAIN_IMG_DIR,
            cls.TRAIN_ANN_DIR,
            cls.VAL_IMG_DIR,
            cls.VAL_ANN_DIR
        ]
        
        missing_paths = []
        for path in paths_to_check:
            if not os.path.exists(path):
                missing_paths.append(path)
        
        if missing_paths:
            print("Warning: The following paths do not exist:")
            for path in missing_paths:
                print(f"  - {path}")
            return False
        
        print("✅ All dataset paths verified!")
        return True
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*50)
        print("CONFIGURATION SUMMARY")
        print("="*50)
        print(f"Training Images: {cls.TRAIN_IMG_DIR}")
        print(f"Training Annotations: {cls.TRAIN_ANN_DIR}")
        print(f"Validation Images: {cls.VAL_IMG_DIR}")
        print(f"Validation Annotations: {cls.VAL_ANN_DIR}")
        print(f"Classes: {cls.CLASSES}")
        print(f"Number of Classes: {cls.NUM_CLASSES}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Input Size: {cls.INPUT_SIZE}x{cls.INPUT_SIZE}")
        print(f"Model Save Path: {cls.MODEL_SAVE_PATH}")
        print("="*50)