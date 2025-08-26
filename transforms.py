# transforms.py
"""
Image transformations and augmentations for RSOD dataset
Includes CLAHE enhancement and various augmentations
"""

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

class CLAHETransform(ImageOnlyTransform):
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE) transform
    Enhances local contrast in images
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8), always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply(self, image, **params):
        # Convert RGB to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(self.clip_limit, self.tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        merged = cv2.merge((cl, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

def get_train_transform():
    """
    Training transformations with augmentations
    """
    return A.Compose([
        CLAHETransform(),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),  # Randomly adjusts brightness and contrast within ±20%
        A.HorizontalFlip(p=0.5),  # 50% chance to flip horizontally
        A.RandomRotate90(p=0.5),  # Randomly rotates by 90°, 180°, or 270°
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_val_transform():
    """
    Validation transformations without augmentations
    """
    return A.Compose([
        CLAHETransform(),
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_inference_transform():
    """
    Simple inference transformations
    """
    return A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])