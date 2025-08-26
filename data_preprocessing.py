# data_preprocessing.py
"""
Data preprocessing utilities for RSOD dataset
Converts YOLO format to Pascal VOC format
"""

import os
import cv2
import xml.etree.ElementTree as ET

def yolo_to_pascal_voc(image_dir, label_dir, output_dir, classes):
    """
    Convert YOLO format annotations to Pascal VOC format
    
    Args:
        image_dir: Directory containing images
        label_dir: Directory containing YOLO format labels
        output_dir: Directory to save Pascal VOC XML files
        classes: List of class names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    converted_count = 0
    
    for img_file in sorted(os.listdir(image_dir)):
        if not img_file.endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(image_dir, img_file)
        label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')
        
        # Read image to get dimensions
        img = cv2.imread(img_path)
        h, w, c = img.shape

        # Create XML annotation
        annotation = ET.Element('annotation')
        ET.SubElement(annotation, 'folder').text = os.path.basename(image_dir)
        ET.SubElement(annotation, 'filename').text = img_file
        ET.SubElement(annotation, 'path').text = img_path

        size = ET.SubElement(annotation, 'size')
        ET.SubElement(size, 'width').text = str(w)
        ET.SubElement(size, 'height').text = str(h)
        ET.SubElement(size, 'depth').text = str(c)

        ET.SubElement(annotation, 'segmented').text = '0'

        # Process YOLO annotations if label file exists
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls_id, xc, yc, bw, bh = map(float, line.strip().split())
                    cls_name = classes[int(cls_id)]
                    
                    # Convert YOLO format to Pascal VOC format
                    x1 = int((xc - bw/2) * w)
                    y1 = int((yc - bh/2) * h)
                    x2 = int((xc + bw/2) * w)
                    y2 = int((yc + bh/2) * h)

                    obj = ET.SubElement(annotation, 'object')
                    ET.SubElement(obj, 'name').text = cls_name
                    ET.SubElement(obj, 'pose').text = 'Unspecified'
                    ET.SubElement(obj, 'truncated').text = '0'
                    ET.SubElement(obj, 'difficult').text = '0'

                    bndbox = ET.SubElement(obj, 'bndbox')
                    ET.SubElement(bndbox, 'xmin').text = str(max(0, x1))
                    ET.SubElement(bndbox, 'ymin').text = str(max(0, y1))
                    ET.SubElement(bndbox, 'xmax').text = str(min(w, x2))
                    ET.SubElement(bndbox, 'ymax').text = str(min(h, y2))

        # Write XML file
        tree = ET.ElementTree(annotation)
        xml_file = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.xml')
        tree.write(xml_file)
        converted_count += 1

    print(f"✅ Converted {converted_count} images to Pascal VOC XMLs!")
    return converted_count

if __name__ == "__main__":
    # Example usage
    image_dir = "/path/to/images"
    label_dir = "/path/to/labels"
    output_dir = "/path/to/annotations"
    classes = ['aircraft', 'oiltank', 'overpass', 'playground']
    
    yolo_to_pascal_voc(image_dir, label_dir, output_dir, classes)