"""
EasiVisi - Inference Utilities
"""
import os
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Cache for loaded models
_model_cache = {}


def load_model(model_path):
    """Load a YOLO model from weights file."""
    global _model_cache
    
    if model_path in _model_cache:
        return _model_cache[model_path]
    
    from ultralytics import YOLO
    model = YOLO(model_path)
    _model_cache[model_path] = model
    
    return model


def clear_model_cache():
    """Clear the model cache."""
    global _model_cache
    _model_cache = {}


def detect(model_path, image_path, conf_threshold=0.25):
    """Run detection on a single image."""
    model = load_model(model_path)
    
    results = model(image_path, conf=conf_threshold)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for i, box in enumerate(boxes):
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x1': int(box.xyxy[0][0]),
                        'y1': int(box.xyxy[0][1]),
                        'x2': int(box.xyxy[0][2]),
                        'y2': int(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
    
    return detections


def detect_from_bytes(model_path, image_bytes, conf_threshold=0.25):
    """Run detection on image bytes."""
    model = load_model(model_path)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    results = model(img, conf=conf_threshold)
    
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                detection = {
                    'class_id': int(box.cls[0]),
                    'class_name': result.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),
                    'bbox': {
                        'x1': int(box.xyxy[0][0]),
                        'y1': int(box.xyxy[0][1]),
                        'x2': int(box.xyxy[0][2]),
                        'y2': int(box.xyxy[0][3])
                    }
                }
                detections.append(detection)
    
    return detections, img.shape[1], img.shape[0]  # detections, width, height


def draw_predictions(image_path, detections, output_path=None):
    """Draw bounding boxes on image."""
    img = cv2.imread(image_path)
    
    # Color palette
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0)
    ]
    
    for det in detections:
        bbox = det['bbox']
        class_id = det['class_id']
        class_name = det.get('class_name', f'Class {class_id}')
        conf = det['confidence']
        
        color = colors[class_id % len(colors)]
        
        # Draw box
        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
        
        # Draw label
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, 
                     (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                     (bbox['x1'] + label_size[0], bbox['y1']),
                     color, -1)
        cv2.putText(img, label, (bbox['x1'], bbox['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
    
    return img


def draw_predictions_bytes(image_bytes, detections):
    """Draw predictions and return as base64 encoded image."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0)
    ]
    
    for det in detections:
        bbox = det['bbox']
        class_id = det['class_id']
        class_name = det.get('class_name', f'Class {class_id}')
        conf = det['confidence']
        
        color = colors[class_id % len(colors)]
        
        cv2.rectangle(img, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), color, 2)
        
        label = f"{class_name}: {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, 
                     (bbox['x1'], bbox['y1'] - label_size[1] - 10),
                     (bbox['x1'] + label_size[0], bbox['y1']),
                     color, -1)
        cv2.putText(img, label, (bbox['x1'], bbox['y1'] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Convert to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return img_base64


def batch_detect(model_path, image_paths, conf_threshold=0.25):
    """Run detection on multiple images."""
    results = []
    
    for img_path in image_paths:
        try:
            detections = detect(model_path, img_path, conf_threshold)
            results.append({
                'image': img_path,
                'detections': detections,
                'success': True
            })
        except Exception as e:
            results.append({
                'image': img_path,
                'error': str(e),
                'success': False
            })
    
    return results


def list_available_models(models_dir, runs_dir):
    """List all available model weights."""
    models = []
    
    # Pre-trained models
    pretrained = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    for model_name in pretrained:
        models.append({
            'name': model_name,
            'type': 'pretrained',
            'path': model_name
        })
    
    # Custom models from runs
    if os.path.exists(runs_dir):
        for project in os.listdir(runs_dir):
            project_path = os.path.join(runs_dir, project)
            if os.path.isdir(project_path):
                for run in os.listdir(project_path):
                    weights_path = os.path.join(project_path, run, 'weights', 'best.pt')
                    if os.path.exists(weights_path):
                        models.append({
                            'name': f"{project}/{run}/best.pt",
                            'type': 'trained',
                            'path': weights_path
                        })
    
    # Models in models directory
    if os.path.exists(models_dir):
        for f in os.listdir(models_dir):
            if f.endswith('.pt'):
                models.append({
                    'name': f,
                    'type': 'custom',
                    'path': os.path.join(models_dir, f)
                })
    
    return models
