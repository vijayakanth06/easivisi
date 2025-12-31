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
    """Split dataset into train/val sets.
    
    Collects all images from the dataset (including from existing train/val folders)
    and redistributes them according to the specified validation ratio.
    """
    base_path = os.path.join(DATASET_DIR, dataset_name)
    images_path = os.path.join(base_path, 'images')
    labels_path = os.path.join(base_path, 'labels')
    
    # Collect ALL images from all possible locations
    all_images = []
    search_paths = [
        images_path,                          # Base images folder
        os.path.join(images_path, 'train'),   # Train folder
        os.path.join(images_path, 'val'),     # Val folder
    ]
    
    extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            for f in os.listdir(search_path):
                file_path = Path(search_path) / f
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    all_images.append(file_path)
    
    if not all_images:
        return {'train': 0, 'val': 0, 'message': 'No images found in dataset'}
    
    # Create directories
    train_img_dir = os.path.join(images_path, 'train')
    val_img_dir = os.path.join(images_path, 'val')
    train_lbl_dir = os.path.join(labels_path, 'train')
    val_lbl_dir = os.path.join(labels_path, 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Shuffle and split
    random.seed(seed)
    random.shuffle(all_images)
    
    val_count = max(1, int(len(all_images) * val_ratio))  # At least 1 for validation
    val_images = all_images[:val_count]
    train_images = all_images[val_count:]
    
    train_moved = 0
    val_moved = 0
    
    # Move to train folder
    for img in train_images:
        if img.exists():
            dest = os.path.join(train_img_dir, img.name)
            if str(img) != dest:  # Only move if not already there
                shutil.move(str(img), dest)
            train_moved += 1
            
            # Move corresponding label if exists (check all possible locations)
            for lbl_search in [labels_path, train_lbl_dir, val_lbl_dir]:
                label_file = os.path.join(lbl_search, img.stem + '.txt')
                if os.path.exists(label_file):
                    lbl_dest = os.path.join(train_lbl_dir, img.stem + '.txt')
                    if label_file != lbl_dest:
                        shutil.move(label_file, lbl_dest)
                    break
    
    # Move to val folder
    for img in val_images:
        if img.exists():
            dest = os.path.join(val_img_dir, img.name)
            if str(img) != dest:  # Only move if not already there
                shutil.move(str(img), dest)
            val_moved += 1
            
            # Move corresponding label if exists
            for lbl_search in [labels_path, train_lbl_dir, val_lbl_dir]:
                label_file = os.path.join(lbl_search, img.stem + '.txt')
                if os.path.exists(label_file):
                    lbl_dest = os.path.join(val_lbl_dir, img.stem + '.txt')
                    if label_file != lbl_dest:
                        shutil.move(label_file, lbl_dest)
                    break
    
    return {
        'train': train_moved,
        'val': val_moved,
        'message': f'Split completed: {train_moved} training, {val_moved} validation'
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
