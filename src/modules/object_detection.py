"""
Object detection module for handling YOLO-based object detection.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from ultralytics import YOLO

from core.config import config


@dataclass
class Detection:
    """Data class for object detection results."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    class_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'center': self.center,
            'class_id': self.class_id
        }


@dataclass
class DetectionResult:
    """Data class for frame detection results."""
    detections: List[Detection]
    frame_count: Dict[str, int]  # class_name -> count
    processing_time: float
    
    def get_person_count(self) -> int:
        """Get number of persons detected."""
        return self.frame_count.get('person', 0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'detections': [d.to_dict() for d in self.detections],
            'frame_count': self.frame_count,
            'processing_time': self.processing_time
        }


class YOLODetector:
    """YOLO-based object detector with configurable settings."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.default_model
        self.model = None
        self.is_loaded = False
        self.confidence_threshold = config.default_confidence_threshold
        self.person_only = False
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load YOLO model."""
        try:
            self.model = YOLO(self.model_name)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            self.is_loaded = False
            return False
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for detections."""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_person_only(self, person_only: bool):
        """Set whether to detect only persons."""
        self.person_only = person_only
    
    def detect_objects(self, frame: cv2.Mat) -> Optional[DetectionResult]:
        """Detect objects in frame."""
        if not self.is_loaded or frame is None:
            return None
        
        try:
            import time
            start_time = time.time()
            
            # Run detection
            results = self.model(frame, stream=True)
            
            detections = []
            frame_count = {}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Extract class and confidence
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Apply filters
                    if confidence < self.confidence_threshold:
                        continue
                    
                    if self.person_only and class_name != 'person':
                        continue
                    
                    # Calculate center point
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # Create detection object
                    detection = Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        class_id=class_id
                    )
                    
                    detections.append(detection)
                    
                    # Update frame count
                    frame_count[class_name] = frame_count.get(class_name, 0) + 1
            
            processing_time = time.time() - start_time
            
            return DetectionResult(
                detections=detections,
                frame_count=frame_count,
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"Error during object detection: {e}")
            return None
    
    def draw_detections(self, frame: cv2.Mat, detections: List[Detection]) -> cv2.Mat:
        """Draw detection boxes and labels on frame."""
        if frame is None or not detections:
            return frame
        
        # Create a copy to avoid modifying original
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Choose color based on class
            color = config.person_color if detection.class_name == 'person' else config.other_object_color
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, config.text_font, config.text_scale, config.text_thickness)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       config.text_font, config.text_scale, (255, 255, 255), config.text_thickness)
        
        return annotated_frame
    
    def get_class_names(self) -> List[str]:
        """Get list of detectable class names."""
        if not self.is_loaded:
            return []
        return list(self.model.names.values())
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_loaded:
            return {}
        
        return {
            'model_name': self.model_name,
            'is_loaded': self.is_loaded,
            'confidence_threshold': self.confidence_threshold,
            'person_only': self.person_only,
            'detectable_classes': self.get_class_names()
        }


class DetectionFilter:
    """Filter for detection results based on various criteria."""
    
    @staticmethod
    def by_confidence(detections: List[Detection], min_confidence: float) -> List[Detection]:
        """Filter detections by minimum confidence."""
        return [d for d in detections if d.confidence >= min_confidence]
    
    @staticmethod
    def by_class_names(detections: List[Detection], class_names: List[str]) -> List[Detection]:
        """Filter detections by class names."""
        return [d for d in detections if d.class_name in class_names]
    
    @staticmethod
    def by_class_ids(detections: List[Detection], class_ids: List[int]) -> List[Detection]:
        """Filter detections by class IDs."""
        return [d for d in detections if d.class_id in class_ids]
    
    @staticmethod
    def by_bbox_size(detections: List[Detection], min_area: int, max_area: int = None) -> List[Detection]:
        """Filter detections by bounding box size."""
        filtered = []
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area and (max_area is None or area <= max_area):
                filtered.append(detection)
        return filtered
    
    @staticmethod
    def by_center_region(detections: List[Detection], region: Tuple[int, int, int, int]) -> List[Detection]:
        """Filter detections by center point within region."""
        rx1, ry1, rx2, ry2 = region
        filtered = []
        for detection in detections:
            cx, cy = detection.center
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                filtered.append(detection)
        return filtered


class DetectionStats:
    """Statistics for detection results."""
    
    def __init__(self):
        self.total_detections = 0
        self.class_counts = {}
        self.confidence_sum = {}
        self.processing_times = []
    
    def update(self, result: DetectionResult):
        """Update statistics with new detection result."""
        self.total_detections += len(result.detections)
        self.processing_times.append(result.processing_time)
        
        for class_name, count in result.frame_count.items():
            self.class_counts[class_name] = self.class_counts.get(class_name, 0) + count
        
        # Calculate confidence sums
        for detection in result.detections:
            if detection.class_name not in self.confidence_sum:
                self.confidence_sum[detection.class_name] = []
            self.confidence_sum[detection.class_name].append(detection.confidence)
    
    def get_average_confidence(self, class_name: str = None) -> float:
        """Get average confidence for a class or all classes."""
        if class_name:
            confidences = self.confidence_sum.get(class_name, [])
            return sum(confidences) / len(confidences) if confidences else 0.0
        else:
            all_confidences = []
            for conf_list in self.confidence_sum.values():
                all_confidences.extend(conf_list)
            return sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
    
    def get_average_processing_time(self) -> float:
        """Get average processing time."""
        return sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get statistics summary."""
        return {
            'total_detections': self.total_detections,
            'class_counts': self.class_counts,
            'average_confidence': self.get_average_confidence(),
            'average_processing_time': self.get_average_processing_time(),
            'class_confidences': {cls: self.get_average_confidence(cls) for cls in self.class_counts}
        }


# Factory function
def create_detector(model_name: str = None) -> YOLODetector:
    """Create and return a configured YOLO detector."""
    return YOLODetector(model_name)
