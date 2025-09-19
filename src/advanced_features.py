"""
Advanced features module for video analysis.
This module has been refactored to use the new modular structure.
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import from new modules
from core.config import config
from modules.object_detection import create_detector
from modules.tracking import create_person_tracker
from modules.database import create_database_manager
from modules.video_processing import VideoProcessor


class AdvancedAnalyzer:
    """Advanced analyzer using the new modular structure."""
    
    def __init__(self):
        # Initialize components using the new modules
        self.detector = create_detector(config.default_model)
        self.tracker = create_person_tracker()
        self.db_manager = create_database_manager()
        self.video_processor = VideoProcessor()
        
        # Legacy compatibility
        self.tracking_objects = {}
        self.object_id_counter = 0
        
        # Note: CSRT tracker is not used in current implementation
        # but kept for potential future use
        
    def track_persons(self, frame):
        """Track persons across frames using the new modular structure"""
        try:
            if frame is None:
                return self.tracking_objects
                
            # Use the new object detection module
            detection_result = self.detector.detect(frame)
            
            # Filter for person detections only
            person_detections = [d for d in detection_result.detections if d.class_name == 'person']
            
            # Use the new tracking module
            tracked_objects = self.tracker.update(person_detections)
            
            # Convert to legacy format for compatibility
            self.tracking_objects = {
                obj_id: {
                    'bbox': [int(obj.bbox.x1), int(obj.bbox.y1), int(obj.bbox.x2), int(obj.bbox.y2)],
                    'center': (int(obj.center.x), int(obj.center.y)),
                    'disappeared': obj.disappeared_count
                }
                for obj_id, obj in tracked_objects.items()
            }
            
            return self.tracking_objects
        except Exception as e:
            print(f"Error in person tracking: {e}")
            return self.tracking_objects
    
    def update_tracking(self, detections):
        """Update object tracking"""
        try:
            # Remove old objects that haven't been seen for a while
            current_time = datetime.now()
            objects_to_remove = []
            
            for obj_id, obj_data in self.tracking_objects.items():
                time_diff = (current_time - obj_data['last_seen']).total_seconds()
                if time_diff > 5.0:  # Remove objects not seen for 5 seconds
                    objects_to_remove.append(obj_id)
            
            for obj_id in objects_to_remove:
                del self.tracking_objects[obj_id]
            
            # Update tracking with new detections
            for detection in detections:
                x1, y1, x2, y2 = detection
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Find closest tracked object
                closest_id = None
                min_distance = float('inf')
                
                for obj_id, obj_data in self.tracking_objects.items():
                    obj_center = obj_data['center']
                    distance = np.sqrt((center[0] - obj_center[0])**2 + 
                                     (center[1] - obj_center[1])**2)
                    
                    if distance < min_distance and distance < 50:  # Threshold
                        min_distance = distance
                        closest_id = obj_id
                
                if closest_id is not None:
                    self.tracking_objects[closest_id].update({
                        'center': center,
                        'bbox': (x1, y1, x2, y2),
                        'last_seen': current_time
                    })
                else:
                    # New object
                    self.object_id_counter += 1
                    self.tracking_objects[self.object_id_counter] = {
                        'center': center,
                        'bbox': (x1, y1, x2, y2),
                        'first_seen': current_time,
                        'last_seen': current_time
                    }
        except Exception as e:
            print(f"Error in tracking update: {e}")
    
    def generate_report(self, video_path):
        """Generate analysis report"""
        report = {
            'video_path': video_path,
            'analysis_date': datetime.now().isoformat(),
            'total_persons_detected': len(self.tracking_objects),
            'person_tracking': {}
        }
        
        for obj_id, obj_data in self.tracking_objects.items():
            duration = obj_data['last_seen'] - obj_data['first_seen']
            report['person_tracking'][f'person_{obj_id}'] = {
                'first_seen': obj_data['first_seen'].isoformat(),
                'last_seen': obj_data['last_seen'].isoformat(),
                'duration_seconds': duration.total_seconds()
            }
        
        return report

class DatabaseManager:
    def __init__(self, db_path='video_analysis.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                analysis_date TEXT,
                total_frames INTEGER,
                persons_detected INTEGER,
                report_json TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS frame_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER,
                frame_number INTEGER,
                timestamp REAL,
                detections_json TEXT,
                FOREIGN KEY (session_id) REFERENCES analysis_sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_session(self, video_path, total_frames, persons_detected, report):
        """Save analysis session to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO analysis_sessions 
                (video_path, analysis_date, total_frames, persons_detected, report_json)
                VALUES (?, ?, ?, ?, ?)
            ''', (video_path, datetime.now().isoformat(), total_frames, 
                  persons_detected, json.dumps(report)))
            
            session_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return session_id
        except Exception as e:
            print(f"Error saving session to database: {e}")
            return None
    
    def save_frame_detection(self, session_id, frame_number, timestamp, detections):
        """Save frame detection data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO frame_detections 
                (session_id, frame_number, timestamp, detections_json)
                VALUES (?, ?, ?, ?)
            ''', (session_id, frame_number, timestamp, json.dumps(detections)))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error saving frame detection: {e}")
    
    def get_analysis_sessions(self):
        """Get all analysis sessions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM analysis_sessions ORDER BY analysis_date DESC')
            sessions = cursor.fetchall()
            
            conn.close()
            return sessions
        except Exception as e:
            print(f"Error getting analysis sessions: {e}")
            return []