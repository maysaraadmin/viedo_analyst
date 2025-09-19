"""
Tracking module for person tracking across video frames.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from core.config import config
from modules.object_detection import Detection


@dataclass
class TrackedObject:
    """Data class for tracked object information."""
    object_id: int
    class_name: str
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    first_seen: datetime
    last_seen: datetime
    confidence: float
    trajectory: List[Tuple[int, int]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'object_id': self.object_id,
            'class_name': self.class_name,
            'center': self.center,
            'bbox': self.bbox,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'confidence': self.confidence,
            'trajectory': self.trajectory,
            'trajectory_length': len(self.trajectory)
        }


class PersonTracker:
    """Person tracker using simple centroid-based tracking."""
    
    def __init__(self, max_disappeared: int = None, max_distance: int = None):
        self.max_disappeared = max_disappeared or config.max_disappeared_frames
        self.max_distance = max_distance or config.max_tracking_distance
        
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 1
        self.disappeared_count: Dict[int, int] = {}
        
        # Optional: CSRT tracker for more accurate tracking
        self.use_csrt = config.use_csrt_tracker
        self.csrt_trackers: Dict[int, cv2.Tracker] = {}
    
    def update(self, detections: List[Detection]) -> Dict[int, TrackedObject]:
        """Update tracking with new detections."""
        current_time = datetime.now()
        
        if not detections:
            # No detections, update disappeared count
            for object_id in list(self.disappeared_count.keys()):
                self.disappeared_count[object_id] += 1
                if self.disappeared_count[object_id] > self.max_disappeared:
                    self._remove_object(object_id)
            return self.tracked_objects
        
        # Convert detections to center points
        detection_centers = [(d.center, d) for d in detections]
        
        # If no objects are being tracked, register all detections
        if not self.tracked_objects:
            for center, detection in detection_centers:
                self._register_object(detection, center, current_time)
            return self.tracked_objects
        
        # Try to match existing objects with new detections
        object_ids = list(self.tracked_objects.keys())
        object_centers = [self.tracked_objects[oid].center for oid in object_ids]
        
        # Calculate distance matrix
        distance_matrix = self._calculate_distance_matrix(object_centers, 
                                                         [center for center, _ in detection_centers])
        
        # Find best matches using Hungarian algorithm (simple greedy approach)
        used_detection_indices = set()
        used_object_indices = set()
        
        # Find closest matches
        matches = []
        for _ in range(min(len(object_ids), len(detection_centers))):
            min_distance = float('inf')
            best_match = None
            
            for i, obj_id in enumerate(object_ids):
                if i in used_object_indices:
                    continue
                    
                for j, (center, _) in enumerate(detection_centers):
                    if j in used_detection_indices:
                        continue
                    
                    distance = distance_matrix[i][j]
                    if distance < min_distance and distance < self.max_distance:
                        min_distance = distance
                        best_match = (i, j, distance)
            
            if best_match:
                obj_idx, det_idx, distance = best_match
                matches.append((object_ids[obj_idx], detection_centers[det_idx]))
                used_object_indices.add(obj_idx)
                used_detection_indices.add(det_idx)
        
        # Update matched objects
        for object_id, (center, detection) in matches:
            self._update_object(object_id, detection, center, current_time)
            self.disappeared_count[object_id] = 0
        
        # Register new detections that weren't matched
        for j, (center, detection) in enumerate(detection_centers):
            if j not in used_detection_indices:
                self._register_object(detection, center, current_time)
        
        # Mark unmatched objects as disappeared
        for i, object_id in enumerate(object_ids):
            if i not in used_object_indices:
                self.disappeared_count[object_id] += 1
                if self.disappeared_count[object_id] > self.max_disappeared:
                    self._remove_object(object_id)
        
        return self.tracked_objects
    
    def _calculate_distance_matrix(self, object_centers: List[Tuple[int, int]], 
                                 detection_centers: List[Tuple[int, int]]) -> np.ndarray:
        """Calculate distance matrix between objects and detections."""
        matrix = np.zeros((len(object_centers), len(detection_centers)))
        
        for i, obj_center in enumerate(object_centers):
            for j, det_center in enumerate(detection_centers):
                distance = np.sqrt((obj_center[0] - det_center[0])**2 + 
                                 (obj_center[1] - det_center[1])**2)
                matrix[i][j] = distance
        
        return matrix
    
    def _register_object(self, detection: Detection, center: Tuple[int, int], 
                        current_time: datetime):
        """Register a new tracked object."""
        tracked_object = TrackedObject(
            object_id=self.next_object_id,
            class_name=detection.class_name,
            center=center,
            bbox=detection.bbox,
            first_seen=current_time,
            last_seen=current_time,
            confidence=detection.confidence,
            trajectory=[center]
        )
        
        self.tracked_objects[self.next_object_id] = tracked_object
        self.disappeared_count[self.next_object_id] = 0
        
        # Initialize CSRT tracker if enabled
        if self.use_csrt:
            try:
                tracker = cv2.TrackerCSRT_create()
                # Note: We need the actual frame to initialize the tracker
                # This will be handled in the main application
                self.csrt_trackers[self.next_object_id] = tracker
            except Exception as e:
                print(f"Error creating CSRT tracker: {e}")
        
        self.next_object_id += 1
    
    def _update_object(self, object_id: int, detection: Detection, 
                      center: Tuple[int, int], current_time: datetime):
        """Update an existing tracked object."""
        if object_id not in self.tracked_objects:
            return
        
        tracked_object = self.tracked_objects[object_id]
        
        # Update object information
        tracked_object.center = center
        tracked_object.bbox = detection.bbox
        tracked_object.last_seen = current_time
        tracked_object.confidence = detection.confidence
        
        # Update trajectory
        tracked_object.trajectory.append(center)
        
        # Limit trajectory length to prevent memory issues
        if len(tracked_object.trajectory) > config.max_trajectory_length:
            tracked_object.trajectory = tracked_object.trajectory[-config.max_trajectory_length:]
    
    def _remove_object(self, object_id: int):
        """Remove a tracked object."""
        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]
        if object_id in self.disappeared_count:
            del self.disappeared_count[object_id]
        if object_id in self.csrt_trackers:
            del self.csrt_trackers[object_id]
    
    def get_tracked_objects(self) -> Dict[int, TrackedObject]:
        """Get all currently tracked objects."""
        return self.tracked_objects
    
    def get_person_count(self) -> int:
        """Get number of tracked persons."""
        return len([obj for obj in self.tracked_objects.values() if obj.class_name == 'person'])
    
    def get_object_trajectory(self, object_id: int) -> List[Tuple[int, int]]:
        """Get trajectory of a specific object."""
        if object_id in self.tracked_objects:
            return self.tracked_objects[object_id].trajectory
        return []
    
    def get_object_lifetime(self, object_id: int) -> float:
        """Get lifetime of a specific object in seconds."""
        if object_id in self.tracked_objects:
            obj = self.tracked_objects[object_id]
            lifetime = (obj.last_seen - obj.first_seen).total_seconds()
            return lifetime
        return 0.0
    
    def clear_all(self):
        """Clear all tracked objects."""
        self.tracked_objects.clear()
        self.disappeared_count.clear()
        self.csrt_trackers.clear()
        self.next_object_id = 1
    
    def draw_tracking(self, frame: cv2.Mat) -> cv2.Mat:
        """Draw tracking information on frame."""
        if frame is None:
            return frame
        
        annotated_frame = frame.copy()
        
        for object_id, tracked_object in self.tracked_objects.items():
            x1, y1, x2, y2 = tracked_object.bbox
            cx, cy = tracked_object.center
            
            # Draw bounding box
            color = config.person_color if tracked_object.class_name == 'person' else config.other_object_color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(annotated_frame, (cx, cy), 4, color, -1)
            
            # Draw object ID and info
            lifetime = self.get_object_lifetime(object_id)
            label = f"ID: {object_id} | Lifetime: {lifetime:.1f}s"
            
            # Draw label background
            label_size = cv2.getTextSize(label, config.text_font, config.text_scale, config.text_thickness)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       config.text_font, config.text_scale, (255, 255, 255), config.text_thickness)
            
            # Draw trajectory
            if len(tracked_object.trajectory) > 1:
                trajectory_points = np.array(tracked_object.trajectory, dtype=np.int32)
                cv2.polylines(annotated_frame, [trajectory_points], False, color, 2)
        
        return annotated_frame


class TrackingStats:
    """Statistics for tracking performance."""
    
    def __init__(self):
        self.total_tracked_objects = 0
        self.average_lifetime = 0.0
        self.max_lifetime = 0.0
        self.class_distribution = {}
    
    def update(self, tracked_objects: Dict[int, TrackedObject]):
        """Update statistics with current tracked objects."""
        if not tracked_objects:
            return
        
        self.total_tracked_objects = len(tracked_objects)
        
        lifetimes = []
        class_counts = {}
        
        for tracked_object in tracked_objects.values():
            lifetime = (tracked_object.last_seen - tracked_object.first_seen).total_seconds()
            lifetimes.append(lifetime)
            
            class_name = tracked_object.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if lifetimes:
            self.average_lifetime = sum(lifetimes) / len(lifetimes)
            self.max_lifetime = max(lifetimes)
        
        self.class_distribution = class_counts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tracking statistics summary."""
        return {
            'total_tracked_objects': self.total_tracked_objects,
            'average_lifetime': self.average_lifetime,
            'max_lifetime': self.max_lifetime,
            'class_distribution': self.class_distribution
        }


# Factory function
def create_person_tracker() -> PersonTracker:
    """Create and return a configured person tracker."""
    return PersonTracker()
