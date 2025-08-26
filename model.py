# model.py
"""
RetinaNet model definition and utilities
"""

import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def get_model(num_classes, pretrained=True):
    """
    Create RetinaNet model with custom number of classes
    
    Args:
        num_classes: Number of classes (including background)
        pretrained: Whether to use pretrained weights
    
    Returns:
        RetinaNet model
    """
    model = retinanet_resnet50_fpn_v2(pretrained=pretrained)
    
    # Get input features and anchors from the original classification head
    in_features = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    
    # Replace classification head with custom one
    model.head.classification_head = RetinaNetClassificationHead(
        in_features, num_anchors, num_classes
    )
    
    return model

def save_model(model, path):
    """Save model state dict"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device='cpu'):
    """Load model state dict"""
    model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from {path}")
    return model