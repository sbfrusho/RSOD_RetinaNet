# dataset.py
"""
RSOD Dataset class for loading images and annotations
"""

import os
import torch
from torch.utils.data import Dataset
from glob import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

class RSODDataset(Dataset):
    """
    RSOD Dataset class for object detection
    Loads images and Pascal VOC format annotations
    """
    
    def __init__(self, img_dir, ann_dir, transforms=None):
        """
        Args:
            img_dir: Directory containing images
            ann_dir: Directory containing XML annotations
            transforms: Albumentations transforms
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.img_files = sorted(glob(os.path.join(img_dir, '*.jpg')) + 
                               glob(os.path.join(img_dir, '*.png')))
        
        # Class mapping
        self.class_dict = {"aircraft": 1, "oiltank": 2, "overpass": 3, "playground": 4}
        self.id_to_name = {v: k for k, v in self.class_dict.items()}

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Load annotations
        xml_path = os.path.join(self.ann_dir, 
                               os.path.basename(img_path).replace(".jpg", ".xml").replace(".png", ".xml"))
        
        boxes, labels = self._parse_xml(xml_path)
        
        # Convert to numpy arrays
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4))
        labels = np.array(labels) if labels else np.zeros(0,)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}
        return image, target

    def _parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation"""
        if not os.path.exists(xml_path):
            return [], []
            
        root = ET.parse(xml_path).getroot()
        boxes, labels = [], []
        
        for obj in root.findall('object'):
            cls = obj.find('name').text.lower()
            if cls not in self.class_dict:
                continue
                
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_dict[cls])
            
        return boxes, labels

def collate_fn(batch):
    """Custom collate function for DataLoader"""
    return tuple(zip(*batch))