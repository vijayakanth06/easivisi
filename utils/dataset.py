"""
EasiVisi - Dataset Management Utilities
"""
import os
import shutil
import random
import yaml
from pathlib import Path
from PIL import Image
from config import DATASET_DIR


def validate_image(file_path):
    """Validate that file is a valid image."""
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True, None
    except Exception as e:
        return False, str(e)


def get_image_info(file_path):
    """Get image dimensions and format."""
    try:
        with Image.open(file_path) as img:
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode
            }
    except Exception:
        return None


def create_dataset_structure(dataset_name):
    """Create YOLO dataset directory structure."""
    base_path = os.path.join(DATASET_DIR, dataset_name)
    
    dirs = [
        os.path.join(base_path, 'images', 'train'),
        os.path.join(base_path, 'images', 'val'),
        os.path.join(base_path, 'labels', 'train'),
        os.path.join(base_path, 'labels', 'val'),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    return base_path


def split_dataset(dataset_name, val_ratio=0.2, seed=42):
    """Split dataset into train/val sets."""
    base_path = os.path.join(DATASET_DIR, dataset_name)
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    
    # Check for unsplit images (directly in images/ folder)
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']:
        all_images.extend(Path(images_path).glob(ext))
        all_images.extend(Path(images_path).glob(ext.upper()))
    
    if not all_images:
        # Images might already be in train/val folders
        return {'train': 0, 'val': 0, 'message': 'No images to split or already split'}
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_images)
    
    val_count = int(len(all_images) * val_ratio)
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]
    
    # Create directories
    os.makedirs(os.path.join(images_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(images_path, 'val'), exist_ok=True)
    os.makedirs(os.path.join(labels_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(labels_path, 'val'), exist_ok=True)
    
    # Move files
    for img in train_images:
        dest = os.path.join(images_path, 'train', img.name)
        shutil.move(str(img), dest)
        # Move corresponding label if exists
        label_file = os.path.join(labels_path, img.stem + '.txt')
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(labels_path, 'train', img.stem + '.txt'))
    
    for img in val_images:
        dest = os.path.join(images_path, 'val', img.name)
        shutil.move(str(img), dest)
        label_file = os.path.join(labels_path, img.stem + '.txt')
        if os.path.exists(label_file):
            shutil.move(label_file, os.path.join(labels_path, 'val', img.stem + '.txt'))
    
    return {
        'train': len(train_images),
        'val': len(val_images),
        'message': 'Split completed successfully'
    }


def generate_dataset_yaml(dataset_name, class_names):
    """Generate YOLO dataset.yaml file."""
    base_path = os.path.join(DATASET_DIR, dataset_name)
    
    yaml_content = {
        'path': os.path.abspath(base_path),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    yaml_path = os.path.join(base_path, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def get_dataset_stats(dataset_name):
    """Get statistics for a dataset."""
    base_path = os.path.join(DATASET_DIR, dataset_name)
    
    if not os.path.exists(base_path):
        return None
    
    stats = {
        'name': dataset_name,
        'path': base_path,
        'train_images': 0,
        'val_images': 0,
        'train_labels': 0,
        'val_labels': 0,
        'classes': [],
        'has_yaml': False
    }
    
    # Count images
    train_img_path = os.path.join(base_path, 'images', 'train')
    val_img_path = os.path.join(base_path, 'images', 'val')
    
    if os.path.exists(train_img_path):
        stats['train_images'] = len([f for f in os.listdir(train_img_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))])
    
    if os.path.exists(val_img_path):
        stats['val_images'] = len([f for f in os.listdir(val_img_path) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))])
    
    # Count labels
    train_lbl_path = os.path.join(base_path, 'labels', 'train')
    val_lbl_path = os.path.join(base_path, 'labels', 'val')
    
    if os.path.exists(train_lbl_path):
        stats['train_labels'] = len([f for f in os.listdir(train_lbl_path) if f.endswith('.txt')])
    
    if os.path.exists(val_lbl_path):
        stats['val_labels'] = len([f for f in os.listdir(val_lbl_path) if f.endswith('.txt')])
    
    # Check for dataset.yaml
    yaml_path = os.path.join(base_path, 'dataset.yaml')
    if os.path.exists(yaml_path):
        stats['has_yaml'] = True
        try:
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
                if 'names' in yaml_data:
                    if isinstance(yaml_data['names'], dict):
                        stats['classes'] = list(yaml_data['names'].values())
                    else:
                        stats['classes'] = yaml_data['names']
        except Exception:
            pass
    
    return stats


def list_datasets():
    """List all available datasets."""
    datasets = []
    
    if not os.path.exists(DATASET_DIR):
        return datasets
    
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path):
            stats = get_dataset_stats(item)
            if stats:
                datasets.append(stats)
    
    return datasets


def delete_dataset(dataset_name):
    """Delete a dataset."""
    base_path = os.path.join(DATASET_DIR, dataset_name)
    
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
        return True
    return False
