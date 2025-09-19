"""
Face detection and person profiling module.
This module handles face detection from person bounding boxes and creates person profiles.
"""

import cv2
import numpy as np
import os
import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib

from core.config import config
from modules.object_detection import Detection
from modules.database import PersonProfile as DatabasePersonProfile


@dataclass
class FaceDetection:
    """Data class for face detection results."""
    face_id: str
    person_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    session_id: int = 0
    frame_number: int = 0
    confidence: float = 0.0
    image_path: str = ""
    timestamp: datetime = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'face_id': self.face_id,
            'person_id': self.person_id,
            'session_id': self.session_id,
            'frame_number': self.frame_number,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'image_path': self.image_path,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class PersonProfile:
    """Data class for person profile information."""
    person_id: int
    profile_name: str
    first_seen: datetime
    last_seen: datetime
    total_appearances: int
    face_images: List[str]
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'person_id': self.person_id,
            'profile_name': self.profile_name,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat(),
            'total_appearances': self.total_appearances,
            'face_images': self.face_images,
            'description': self.description
        }


class FaceDetector:
    """Face detector using OpenCV's Haar cascade or DNN-based face detection."""
    
    def __init__(self, detection_method: str = 'haar'):
        """
        Initialize face detector.
        
        Args:
            detection_method: 'haar' for Haar cascade, 'dnn' for DNN-based detection
        """
        self.detection_method = detection_method
        self.face_cascade = None
        self.net = None
        self._load_model()
    
    def _load_model(self):
        """Load face detection model."""
        try:
            if self.detection_method == 'haar':
                # Load Haar cascade classifier
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                if self.face_cascade.empty():
                    print("Warning: Could not load Haar cascade classifier")
            elif self.detection_method == 'dnn':
                # Load DNN model
                model_file = "opencv_face_detector_uint8.pb"
                config_file = "opencv_face_detector.pbtxt"
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', model_file)
                config_path = os.path.join(os.path.dirname(__file__), '..', 'models', config_file)
                
                if os.path.exists(model_path) and os.path.exists(config_path):
                    self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
                else:
                    print("Warning: DNN model files not found, falling back to Haar cascade")
                    self.detection_method = 'haar'
                    self._load_model()
        except Exception as e:
            print(f"Error loading face detection model: {e}")
    
    def detect_faces_in_person(self, frame: np.ndarray, person_detection: Detection) -> List[FaceDetection]:
        """
        Detect faces within a person's bounding box.
        
        Args:
            frame: Full frame image
            person_detection: Person detection result
            
        Returns:
            List of face detections
        """
        if person_detection.class_name != 'person':
            return []
        
        # Extract person region
        x1, y1, x2, y2 = person_detection.bbox
        person_region = frame[y1:y2, x1:x2]
        
        if person_region.size == 0:
            return []
        
        # Convert to grayscale for face detection
        gray_person = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        faces = []
        
        if self.detection_method == 'haar' and self.face_cascade is not None:
            # Detect faces using Haar cascade
            face_rects = self.face_cascade.detectMultiScale(
                gray_person,
                scaleFactor=1.1,
                minNeighbors=3,  # Reduced from 5
                minSize=(20, 20)  # Reduced from 30
            )
            
            print(f"DEBUG: Found {len(face_rects)} faces")
            
            for (fx, fy, fw, fh) in face_rects:
                # Convert face coordinates back to full frame coordinates
                face_x1 = x1 + fx
                face_y1 = y1 + fy
                face_x2 = face_x1 + fw
                face_y2 = face_y1 + fh
                
                # Generate unique face ID
                face_id = self._generate_face_id(person_detection, (face_x1, face_y1, face_x2, face_y2))
                
                face_detection = FaceDetection(
                    face_id=face_id,
                    person_id=person_detection.class_id,  # Using class_id as temporary person_id
                    bbox=(face_x1, face_y1, face_x2, face_y2),
                    confidence=0.8,  # Default confidence for Haar cascade
                    image_path="",  # Will be set when saving
                    timestamp=datetime.now()
                )
                faces.append(face_detection)
        
        elif self.detection_method == 'dnn' and self.net is not None:
            # Detect faces using DNN
            blob = cv2.dnn.blobFromImage(person_region, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            h, w = person_region.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    x1_det = int(detections[0, 0, i, 3] * w)
                    y1_det = int(detections[0, 0, i, 4] * h)
                    x2_det = int(detections[0, 0, i, 5] * w)
                    y2_det = int(detections[0, 0, i, 6] * h)
                    
                    # Convert face coordinates back to full frame coordinates
                    face_x1 = x1 + x1_det
                    face_y1 = y1 + y1_det
                    face_x2 = x1 + x2_det
                    face_y2 = y1 + y2_det
                    
                    # Generate unique face ID
                    face_id = self._generate_face_id(person_detection, (face_x1, face_y1, face_x2, face_y2))
                    
                    face_detection = FaceDetection(
                        face_id=face_id,
                        person_id=person_detection.class_id,
                        bbox=(face_x1, face_y1, face_x2, face_y2),
                        confidence=float(confidence),
                        image_path="",
                        timestamp=datetime.now()
                    )
                    faces.append(face_detection)
        
        return faces
    
    def detect_faces(self, person_region: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a person region image.
        
        Args:
            person_region: Person region image (cropped from full frame)
            
        Returns:
            List of face detections with coordinates relative to the person region
        """
        if person_region.size == 0:
            return []
        
        # Convert to grayscale for face detection
        gray_person = cv2.cvtColor(person_region, cv2.COLOR_BGR2GRAY)
        
        faces = []
        
        if self.detection_method == 'haar' and self.face_cascade is not None:
            # Detect faces using Haar cascade
            face_rects = self.face_cascade.detectMultiScale(
                gray_person,
                scaleFactor=1.1,
                minNeighbors=3,  # Reduced from 5
                minSize=(20, 20)  # Reduced from 30
            )
            
            print(f"DEBUG: Found {len(face_rects)} faces")
            
            for (fx, fy, fw, fh) in face_rects:
                # Face coordinates relative to person region
                face_x1 = fx
                face_y1 = fy
                face_x2 = fx + fw
                face_y2 = fy + fh
                
                # Generate unique face ID
                face_id = hashlib.md5(f"{face_x1}_{face_y1}_{face_x2}_{face_y2}_{datetime.now().timestamp()}".encode()).hexdigest()[:16]
                
                face_detection = FaceDetection(
                    face_id=face_id,
                    person_id=0,  # Will be set by caller
                    bbox=(face_x1, face_y1, face_x2, face_y2),
                    confidence=0.8,  # Default confidence for Haar cascade
                    image_path="",  # Will be set when saving
                    timestamp=datetime.now()
                )
                faces.append(face_detection)
        
        elif self.detection_method == 'dnn' and self.net is not None:
            # Detect faces using DNN
            blob = cv2.dnn.blobFromImage(person_region, 1.0, (300, 300), [104, 117, 123], False, False)
            self.net.setInput(blob)
            detections = self.net.forward()
            
            h, w = person_region.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    x1_det = int(detections[0, 0, i, 3] * w)
                    y1_det = int(detections[0, 0, i, 4] * h)
                    x2_det = int(detections[0, 0, i, 5] * w)
                    y2_det = int(detections[0, 0, i, 6] * h)
                    
                    # Generate unique face ID
                    face_id = hashlib.md5(f"{x1_det}_{y1_det}_{x2_det}_{y2_det}_{datetime.now().timestamp()}".encode()).hexdigest()[:16]
                    
                    face_detection = FaceDetection(
                        face_id=face_id,
                        person_id=0,  # Will be set by caller
                        bbox=(x1_det, y1_det, x2_det, y2_det),
                        confidence=float(confidence),
                        image_path="",
                        timestamp=datetime.now()
                    )
                    faces.append(face_detection)
        
        return faces
    
    def _generate_face_id(self, person_detection: Detection, face_bbox: Tuple[int, int, int, int]) -> str:
        """Generate unique face ID based on person detection and face position."""
        # Create a hash based on person position and face position
        data = f"{person_detection.bbox}_{face_bbox}_{datetime.now().timestamp()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def extract_face_image(self, frame: np.ndarray, face_detection: FaceDetection) -> np.ndarray:
        """
        Extract face image from frame based on face detection.
        
        Args:
            frame: Full frame image
            face_detection: Face detection result
            
        Returns:
            Face image
        """
        x1, y1, x2, y2 = face_detection.bbox
        face_image = frame[y1:y2, x1:x2]
        return face_image
    
    def save_face_image(self, face_image: np.ndarray, face_detection: FaceDetection) -> str:
        """
        Save face image to disk.
        
        Args:
            face_image: Face image to save
            face_detection: Face detection result
            
        Returns:
            Path to saved image
        """
        # Create faces directory if it doesn't exist
        faces_dir = os.path.join(config.output_directory, 'faces')
        os.makedirs(faces_dir, exist_ok=True)
        
        # Generate filename
        timestamp = face_detection.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"face_{face_detection.face_id}_{timestamp}.jpg"
        image_path = os.path.join(faces_dir, filename)
        
        # Save image
        cv2.imwrite(image_path, face_image)
        
        return image_path
    
    def capture_and_save_face_screenshot(self, frame: np.ndarray, person_detection: Detection, 
                                       person_id: int, session_id: int, frame_number: int) -> Optional[FaceDetection]:
        """
        Capture face screenshot from person detection and save it.
        
        Args:
            frame: Full frame image
            person_detection: Person detection result
            person_id: ID of the person
            session_id: ID of the current session
            frame_number: Current frame number
            
        Returns:
            FaceDetection object if successful, None otherwise
        """
        # Detect faces in person region
        faces = self.detect_faces_in_person(frame, person_detection)
        
        if not faces:
            return None
        
        # Use the first detected face (most confident)
        face_detection = faces[0]
        
        # Update person_id and other fields
        face_detection.person_id = person_id
        face_detection.session_id = session_id
        face_detection.frame_number = frame_number
        
        # Extract face image
        face_image = self.extract_face_image(frame, face_detection)
        
        if face_image.size == 0:
            return None
        
        # Enhance face image quality
        enhanced_face = self._enhance_face_image(face_image)
        
        # Save face image
        image_path = self.save_face_image(enhanced_face, face_detection)
        face_detection.image_path = image_path
        
        return face_detection
    
    def _enhance_face_image(self, face_image: np.ndarray) -> np.ndarray:
        """
        Enhance face image quality for better storage.
        
        Args:
            face_image: Original face image
            
        Returns:
            Enhanced face image
        """
        # Resize to standard size if needed
        h, w = face_image.shape[:2]
        if h < 50 or w < 50:
            # Too small, upscale
            scale_factor = max(50 / h, 50 / w)
            new_size = (int(w * scale_factor), int(h * scale_factor))
            face_image = cv2.resize(face_image, new_size, interpolation=cv2.INTER_LANCZOS4)
        
        # Apply slight sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) / 9.0
        sharpened = cv2.filter2D(face_image, -1, kernel)
        
        # Blend with original to avoid over-sharpening
        enhanced = cv2.addWeighted(face_image, 0.7, sharpened, 0.3, 0)
        
        # Apply histogram equalization for better contrast
        if len(enhanced.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            # Grayscale image
            enhanced = cv2.equalizeHist(enhanced)
        
        return enhanced
    
    def create_face_thumbnail(self, face_image: np.ndarray, size: Tuple[int, int] = (100, 100)) -> np.ndarray:
        """
        Create a thumbnail version of the face image.
        
        Args:
            face_image: Face image
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail image
        """
        return cv2.resize(face_image, size, interpolation=cv2.INTER_AREA)
    
    def validate_face_detection(self, face_detection: FaceDetection, frame_shape: Tuple[int, int]) -> bool:
        """
        Validate face detection results.
        
        Args:
            face_detection: Face detection to validate
            frame_shape: Shape of the original frame (height, width)
            
        Returns:
            True if valid, False otherwise
        """
        x1, y1, x2, y2 = face_detection.bbox
        h, w = frame_shape
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # Check minimum size
        if (x2 - x1) < 20 or (y2 - y1) < 20:
            return False
        
        # Check aspect ratio (faces are typically not too extreme)
        aspect_ratio = (x2 - x1) / (y2 - y1)
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False
        
        return True


class PersonProfileManager:
    """Manager for creating and maintaining person profiles."""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.profiles: Dict[int, DatabasePersonProfile] = {}
        self.next_profile_id = 1
        self.face_detector = FaceDetector()
    
    def create_or_update_profile(self, person_id: int, face_image: np.ndarray, frame_number: int, face_bbox: Tuple[int, int, int, int]) -> Optional[DatabasePersonProfile]:
        """
        Create or update person profile with face detection.
        
        Args:
            person_id: ID of the person
            face_image: Face image as numpy array
            frame_number: Frame number where face was detected
            face_bbox: Face bounding box (x1, y1, x2, y2)
            
        Returns:
            Updated person profile or None if failed
        """
        try:
            current_time = datetime.now()
            
            # Save face image
            face_image_path = self._save_face_image(face_image, person_id, frame_number)
            
            if not face_image_path:
                return None
            
            # Check if profile already exists in database
            existing_profile = self.db_manager.get_person_profile(person_id)
            
            if existing_profile:
                # Update existing profile
                existing_profile.last_seen = current_time.isoformat()
                existing_profile.total_appearances += 1
                
                # Add new face image to the list
                face_images = json.loads(existing_profile.face_images) if existing_profile.face_images else []
                if face_image_path not in face_images:
                    face_images.append(face_image_path)
                existing_profile.face_images = json.dumps(face_images)
                
                # Update in database
                self.db_manager.update_person_profile(existing_profile.person_id, 
                                                   last_seen=existing_profile.last_seen,
                                                   total_appearances=existing_profile.total_appearances,
                                                   face_images=existing_profile.face_images)
                return existing_profile
            else:
                # Create new profile
                new_profile = DatabasePersonProfile(
                    person_id=person_id,
                    profile_name=f"Person_{person_id}",
                    first_seen=current_time.isoformat(),
                    last_seen=current_time.isoformat(),
                    total_appearances=1,
                    face_images=json.dumps([face_image_path]),
                    description=""
                )
                
                # Save to database
                profile_id = self.db_manager.save_person_profile(new_profile)
                if profile_id:
                    new_profile.person_id = profile_id
                    return new_profile
                else:
                    return None
                
        except Exception as e:
            print(f"Error creating/updating profile: {e}")
            return None
    
    def _save_face_image(self, face_image: np.ndarray, person_id: int, frame_number: int) -> Optional[str]:
        """
        Save face image to disk.
        
        Args:
            face_image: Face image as numpy array
            person_id: ID of the person
            frame_number: Frame number
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            # Create faces directory if it doesn't exist
            faces_dir = os.path.join(config.output_directory, 'faces')
            os.makedirs(faces_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_{person_id}_{frame_number}_{timestamp}.jpg"
            image_path = os.path.join(faces_dir, filename)
            
            # Save image
            cv2.imwrite(image_path, face_image)
            
            return image_path
        except Exception as e:
            print(f"Error saving face image: {e}")
            return None
    
    def get_profile(self, person_id: int) -> Optional[PersonProfile]:
        """Get person profile by ID."""
        return self.profiles.get(person_id)
    
    def get_all_profiles(self) -> List[PersonProfile]:
        """Get all person profiles."""
        return list(self.profiles.values())
    
    def update_profile_name(self, person_id: int, name: str) -> bool:
        """Update person profile name."""
        if person_id in self.profiles:
            self.profiles[person_id].profile_name = name
            return True
        return False
    
    def update_profile_description(self, person_id: int, description: str) -> bool:
        """Update person profile description."""
        if person_id in self.profiles:
            self.profiles[person_id].description = description
            return True
        return False
