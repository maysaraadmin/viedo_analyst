"""
Database module for handling data persistence and analysis session management.
"""

import sqlite3
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from core.config import config


@dataclass
class StoredVideo:
    """Data class for stored video information."""
    video_id: Optional[int] = None
    original_path: str = ""
    stored_path: str = ""
    filename: str = ""
    file_size: int = 0
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    storage_date: str = ""
    last_accessed: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoredVideo':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class AnalysisSession:
    """Data class for analysis session information."""
    session_id: Optional[int] = None
    video_path: str = ""
    analysis_date: str = ""
    total_frames: int = 0
    persons_detected: int = 0
    report_json: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalysisSession':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FrameDetection:
    """Data class for frame detection data."""
    detection_id: Optional[int] = None
    session_id: int = 0
    frame_number: int = 0
    timestamp: float = 0.0
    detections_json: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameDetection':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class FaceDetection:
    """Data class for face detection results."""
    face_id: Optional[int] = None
    person_id: int = 0
    session_id: int = 0
    frame_number: int = 0
    bbox: str = ""  # JSON string of (x1, y1, x2, y2)
    confidence: float = 0.0
    image_path: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FaceDetection':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PersonProfile:
    """Data class for person profile information."""
    person_id: Optional[int] = None
    profile_name: str = ""
    first_seen: str = ""
    last_seen: str = ""
    total_appearances: int = 0
    face_images: str = ""  # JSON string of image paths
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonProfile':
        """Create from dictionary."""
        return cls(**data)


class DatabaseManager:
    """Database manager for SQLite operations."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or config.db_path
        self._ensure_database_directory()
        self._initialize_database()
    
    def _ensure_database_directory(self):
        """Ensure database directory exists."""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
    
    def _initialize_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create analysis_sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_sessions (
                        session_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_path TEXT NOT NULL,
                        analysis_date TEXT NOT NULL,
                        total_frames INTEGER DEFAULT 0,
                        persons_detected INTEGER DEFAULT 0,
                        report_json TEXT
                    )
                ''')
                
                # Create frame_detections table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS frame_detections (
                        detection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        frame_number INTEGER NOT NULL,
                        timestamp REAL NOT NULL,
                        detections_json TEXT,
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
                    )
                ''')
                
                # Create tracked_objects table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tracked_objects (
                        object_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER NOT NULL,
                        object_class TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        trajectory_json TEXT,
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
                    )
                ''')
                
                # Create stored_videos table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS stored_videos (
                        video_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_path TEXT NOT NULL,
                        stored_path TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        file_size INTEGER DEFAULT 0,
                        duration REAL DEFAULT 0.0,
                        width INTEGER DEFAULT 0,
                        height INTEGER DEFAULT 0,
                        fps REAL DEFAULT 0.0,
                        frame_count INTEGER DEFAULT 0,
                        storage_date TEXT NOT NULL,
                        last_accessed TEXT NOT NULL
                    )
                ''')
                
                # Create person_profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS person_profiles (
                        person_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        profile_name TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        total_appearances INTEGER DEFAULT 0,
                        face_images TEXT,  -- JSON array of image paths
                        description TEXT
                    )
                ''')
                
                # Create face_detections table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS face_detections (
                        face_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        person_id INTEGER NOT NULL,
                        session_id INTEGER NOT NULL,
                        frame_number INTEGER NOT NULL,
                        bbox TEXT NOT NULL,  -- JSON string of (x1, y1, x2, y2)
                        confidence REAL DEFAULT 0.0,
                        image_path TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        FOREIGN KEY (person_id) REFERENCES person_profiles (person_id),
                        FOREIGN KEY (session_id) REFERENCES analysis_sessions (session_id)
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            print(f"Error initializing database: {e}")
    
    def save_session(self, session: AnalysisSession) -> Optional[int]:
        """Save analysis session to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_sessions 
                    (video_path, analysis_date, total_frames, persons_detected, report_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session.video_path,
                    session.analysis_date,
                    session.total_frames,
                    session.persons_detected,
                    session.report_json
                ))
                
                session_id = cursor.lastrowid
                conn.commit()
                
                return session_id
                
        except Exception as e:
            print(f"Error saving session to database: {e}")
            return None
    
    def get_session(self, session_id: int) -> Optional[AnalysisSession]:
        """Get analysis session by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT session_id, video_path, analysis_date, total_frames, 
                           persons_detected, report_json
                    FROM analysis_sessions WHERE session_id = ?
                ''', (session_id,))
                
                row = cursor.fetchone()
                if row:
                    return AnalysisSession(
                        session_id=row[0],
                        video_path=row[1],
                        analysis_date=row[2],
                        total_frames=row[3],
                        persons_detected=row[4],
                        report_json=row[5]
                    )
                
                return None
                
        except Exception as e:
            print(f"Error getting session from database: {e}")
            return None
    
    def get_all_sessions(self) -> List[AnalysisSession]:
        """Get all analysis sessions from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT session_id, video_path, analysis_date, total_frames, 
                           persons_detected, report_json
                    FROM analysis_sessions ORDER BY analysis_date DESC
                ''')
                
                sessions = []
                for row in cursor.fetchall():
                    session = AnalysisSession(
                        session_id=row[0],
                        video_path=row[1],
                        analysis_date=row[2],
                        total_frames=row[3],
                        persons_detected=row[4],
                        report_json=row[5]
                    )
                    sessions.append(session)
                
                return sessions
                
        except Exception as e:
            print(f"Error getting sessions from database: {e}")
            return []
    
    def update_session(self, session: AnalysisSession) -> bool:
        """Update analysis session in database."""
        if session.session_id is None:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE analysis_sessions 
                    SET video_path = ?, analysis_date = ?, total_frames = ?,
                        persons_detected = ?, report_json = ?
                    WHERE session_id = ?
                ''', (
                    session.video_path,
                    session.analysis_date,
                    session.total_frames,
                    session.persons_detected,
                    session.report_json,
                    session.session_id
                ))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating session in database: {e}")
            return False
    
    def delete_session(self, session_id: int) -> bool:
        """Delete analysis session and related data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete frame detections
                cursor.execute('DELETE FROM frame_detections WHERE session_id = ?', (session_id,))
                
                # Delete tracked objects
                cursor.execute('DELETE FROM tracked_objects WHERE session_id = ?', (session_id,))
                
                # Delete session
                cursor.execute('DELETE FROM analysis_sessions WHERE session_id = ?', (session_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting session from database: {e}")
            return False
    
    def save_frame_detection(self, frame_detection: FrameDetection) -> Optional[int]:
        """Save frame detection data to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO frame_detections 
                    (session_id, frame_number, timestamp, detections_json)
                    VALUES (?, ?, ?, ?)
                ''', (
                    frame_detection.session_id,
                    frame_detection.frame_number,
                    frame_detection.timestamp,
                    frame_detection.detections_json
                ))
                
                detection_id = cursor.lastrowid
                conn.commit()
                
                return detection_id
                
        except Exception as e:
            print(f"Error saving frame detection to database: {e}")
            return None
    
    def get_frame_detections(self, session_id: int) -> List[FrameDetection]:
        """Get all frame detections for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT detection_id, session_id, frame_number, timestamp, detections_json
                    FROM frame_detections WHERE session_id = ?
                    ORDER BY frame_number
                ''', (session_id,))
                
                detections = []
                for row in cursor.fetchall():
                    detection = FrameDetection(
                        detection_id=row[0],
                        session_id=row[1],
                        frame_number=row[2],
                        timestamp=row[3],
                        detections_json=row[4]
                    )
                    detections.append(detection)
                
                return detections
                
        except Exception as e:
            print(f"Error getting frame detections from database: {e}")
            return []
    
    def save_tracked_object(self, session_id: int, object_class: str, 
                           first_seen: datetime, last_seen: datetime, 
                           trajectory: List[Tuple[int, int]]) -> Optional[int]:
        """Save tracked object data to database."""
        try:
            trajectory_json = json.dumps(trajectory)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO tracked_objects 
                    (session_id, object_class, first_seen, last_seen, trajectory_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    session_id,
                    object_class,
                    first_seen.isoformat(),
                    last_seen.isoformat(),
                    trajectory_json
                ))
                
                object_id = cursor.lastrowid
                conn.commit()
                
                return object_id
                
        except Exception as e:
            print(f"Error saving tracked object to database: {e}")
            return None
    
    def get_tracked_objects(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all tracked objects for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT object_id, object_class, first_seen, last_seen, trajectory_json
                    FROM tracked_objects WHERE session_id = ?
                    ORDER BY first_seen
                ''', (session_id,))
                
                objects = []
                for row in cursor.fetchall():
                    obj_data = {
                        'object_id': row[0],
                        'object_class': row[1],
                        'first_seen': row[2],
                        'last_seen': row[3],
                        'trajectory': json.loads(row[4]) if row[4] else []
                    }
                    objects.append(obj_data)
                
                return objects
                
        except Exception as e:
            print(f"Error getting tracked objects from database: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get session count
                cursor.execute('SELECT COUNT(*) FROM analysis_sessions')
                session_count = cursor.fetchone()[0]
                
                # Get frame detection count
                cursor.execute('SELECT COUNT(*) FROM frame_detections')
                frame_count = cursor.fetchone()[0]
                
                # Get tracked object count
                cursor.execute('SELECT COUNT(*) FROM tracked_objects')
                object_count = cursor.fetchone()[0]
                
                # Get total persons detected
                cursor.execute('SELECT SUM(persons_detected) FROM analysis_sessions')
                total_persons = cursor.fetchone()[0] or 0
                
                # Get database file size
                db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
                
                return {
                    'session_count': session_count,
                    'frame_detection_count': frame_count,
                    'tracked_object_count': object_count,
                    'total_persons_detected': total_persons,
                    'database_size_bytes': db_size,
                    'database_size_mb': round(db_size / (1024 * 1024), 2)
                }
                
        except Exception as e:
            print(f"Error getting database stats: {e}")
            return {}
    
    def cleanup_old_sessions(self, days_to_keep: int = 30) -> int:
        """Clean up sessions older than specified days."""
        try:
            cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get old session IDs
                cursor.execute('''
                    SELECT session_id FROM analysis_sessions 
                    WHERE strftime('%s', analysis_date) < ?
                ''', (cutoff_date,))
                
                old_session_ids = [row[0] for row in cursor.fetchall()]
                
                # Delete old sessions
                deleted_count = 0
                for session_id in old_session_ids:
                    if self.delete_session(session_id):
                        deleted_count += 1
                
                return deleted_count
                
        except Exception as e:
            print(f"Error cleaning up old sessions: {e}")
            return 0
    
    def export_session_data(self, session_id: int, export_path: str) -> bool:
        """Export session data to JSON file."""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            frame_detections = self.get_frame_detections(session_id)
            tracked_objects = self.get_tracked_objects(session_id)
            
            export_data = {
                'session': session.to_dict(),
                'frame_detections': [fd.to_dict() for fd in frame_detections],
                'tracked_objects': tracked_objects,
                'export_timestamp': datetime.now().isoformat()
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting session data: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create a backup of the database."""
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            return True
        except Exception as e:
            print(f"Error backing up database: {e}")
            return False
    
    # Video Storage Methods
    def save_stored_video(self, video: StoredVideo) -> Optional[int]:
        """Save stored video information to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO stored_videos 
                    (original_path, stored_path, filename, file_size, duration, 
                     width, height, fps, frame_count, storage_date, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    video.original_path,
                    video.stored_path,
                    video.filename,
                    video.file_size,
                    video.duration,
                    video.width,
                    video.height,
                    video.fps,
                    video.frame_count,
                    video.storage_date,
                    video.last_accessed
                ))
                
                video_id = cursor.lastrowid
                conn.commit()
                return video_id
                
        except Exception as e:
            print(f"Error saving stored video to database: {e}")
            return None
    
    def get_stored_video(self, video_id: int) -> Optional[StoredVideo]:
        """Get stored video by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT video_id, original_path, stored_path, filename, file_size,
                           duration, width, height, fps, frame_count, storage_date, last_accessed
                    FROM stored_videos WHERE video_id = ?
                ''', (video_id,))
                
                row = cursor.fetchone()
                if row:
                    return StoredVideo(
                        video_id=row[0],
                        original_path=row[1],
                        stored_path=row[2],
                        filename=row[3],
                        file_size=row[4],
                        duration=row[5],
                        width=row[6],
                        height=row[7],
                        fps=row[8],
                        frame_count=row[9],
                        storage_date=row[10],
                        last_accessed=row[11]
                    )
                return None
                
        except Exception as e:
            print(f"Error getting stored video from database: {e}")
            return None
    
    def get_all_stored_videos(self) -> List[StoredVideo]:
        """Get all stored videos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT video_id, original_path, stored_path, filename, file_size,
                           duration, width, height, fps, frame_count, storage_date, last_accessed
                    FROM stored_videos ORDER BY storage_date DESC
                ''')
                
                videos = []
                for row in cursor.fetchall():
                    video = StoredVideo(
                        video_id=row[0],
                        original_path=row[1],
                        stored_path=row[2],
                        filename=row[3],
                        file_size=row[4],
                        duration=row[5],
                        width=row[6],
                        height=row[7],
                        fps=row[8],
                        frame_count=row[9],
                        storage_date=row[10],
                        last_accessed=row[11]
                    )
                    videos.append(video)
                
                return videos
                
        except Exception as e:
            print(f"Error getting stored videos from database: {e}")
            return []
    
    def delete_stored_video(self, video_id: int) -> bool:
        """Delete stored video from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('DELETE FROM stored_videos WHERE video_id = ?', (video_id,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting stored video from database: {e}")
            return False
    
    def update_video_access_time(self, video_id: int) -> bool:
        """Update last accessed time for a video."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE stored_videos SET last_accessed = ? WHERE video_id = ?
                ''', (datetime.now().isoformat(), video_id))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating video access time: {e}")
            return False
    
    # Person Profile Methods
    def save_person_profile(self, profile: PersonProfile) -> Optional[int]:
        """Save person profile to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO person_profiles 
                    (profile_name, first_seen, last_seen, total_appearances, face_images, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    profile.profile_name,
                    profile.first_seen,
                    profile.last_seen,
                    profile.total_appearances,
                    profile.face_images,
                    profile.description
                ))
                
                profile_id = cursor.lastrowid
                conn.commit()
                return profile_id
                
        except Exception as e:
            print(f"Error saving person profile to database: {e}")
            return None
    
    def get_person_profile(self, person_id: int) -> Optional[PersonProfile]:
        """Get person profile by ID."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT person_id, profile_name, first_seen, last_seen, 
                           total_appearances, face_images, description
                    FROM person_profiles WHERE person_id = ?
                ''', (person_id,))
                
                row = cursor.fetchone()
                if row:
                    return PersonProfile(
                        person_id=row[0],
                        profile_name=row[1],
                        first_seen=row[2],
                        last_seen=row[3],
                        total_appearances=row[4],
                        face_images=row[5],
                        description=row[6]
                    )
                return None
                
        except Exception as e:
            print(f"Error getting person profile from database: {e}")
            return None
    
    def get_all_person_profiles(self) -> List[PersonProfile]:
        """Get all person profiles."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT person_id, profile_name, first_seen, last_seen, 
                           total_appearances, face_images, description
                    FROM person_profiles ORDER BY last_seen DESC
                ''')
                
                profiles = []
                for row in cursor.fetchall():
                    profile = PersonProfile(
                        person_id=row[0],
                        profile_name=row[1],
                        first_seen=row[2],
                        last_seen=row[3],
                        total_appearances=row[4],
                        face_images=row[5],
                        description=row[6]
                    )
                    profiles.append(profile)
                
                return profiles
                
        except Exception as e:
            print(f"Error getting person profiles from database: {e}")
            return []
    
    def update_person_profile(self, person_id: int, **kwargs) -> bool:
        """Update person profile fields."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build dynamic update query
                update_fields = []
                values = []
                
                for field, value in kwargs.items():
                    if field in ['profile_name', 'first_seen', 'last_seen', 'total_appearances', 'face_images', 'description']:
                        update_fields.append(f"{field} = ?")
                        values.append(value)
                
                if not update_fields:
                    return False
                
                query = f"UPDATE person_profiles SET {', '.join(update_fields)} WHERE person_id = ?"
                values.append(person_id)
                
                cursor.execute(query, values)
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error updating person profile: {e}")
            return False
    
    def delete_person_profile(self, person_id: int) -> bool:
        """Delete person profile and associated face detections."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete associated face detections first
                cursor.execute('DELETE FROM face_detections WHERE person_id = ?', (person_id,))
                
                # Delete person profile
                cursor.execute('DELETE FROM person_profiles WHERE person_id = ?', (person_id,))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Error deleting person profile: {e}")
            return False
    
    # Face Detection Methods
    def save_face_detection(self, face_detection: FaceDetection) -> Optional[int]:
        """Save face detection to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO face_detections 
                    (person_id, session_id, frame_number, bbox, confidence, image_path, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    face_detection.person_id,
                    face_detection.session_id,
                    face_detection.frame_number,
                    face_detection.bbox,
                    face_detection.confidence,
                    face_detection.image_path,
                    face_detection.timestamp
                ))
                
                face_id = cursor.lastrowid
                conn.commit()
                return face_id
                
        except Exception as e:
            print(f"Error saving face detection to database: {e}")
            return None
    
    def get_face_detections_for_person(self, person_id: int) -> List[FaceDetection]:
        """Get all face detections for a person."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT face_id, person_id, session_id, frame_number, bbox, 
                           confidence, image_path, timestamp
                    FROM face_detections WHERE person_id = ?
                    ORDER BY timestamp DESC
                ''', (person_id,))
                
                detections = []
                for row in cursor.fetchall():
                    detection = FaceDetection(
                        face_id=row[0],
                        person_id=row[1],
                        session_id=row[2],
                        frame_number=row[3],
                        bbox=row[4],
                        confidence=row[5],
                        image_path=row[6],
                        timestamp=row[7]
                    )
                    detections.append(detection)
                
                return detections
                
        except Exception as e:
            print(f"Error getting face detections from database: {e}")
            return []
    
    def get_face_detections_for_session(self, session_id: int) -> List[FaceDetection]:
        """Get all face detections for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT face_id, person_id, session_id, frame_number, bbox, 
                           confidence, image_path, timestamp
                    FROM face_detections WHERE session_id = ?
                    ORDER BY frame_number
                ''', (session_id,))
                
                detections = []
                for row in cursor.fetchall():
                    detection = FaceDetection(
                        face_id=row[0],
                        person_id=row[1],
                        session_id=row[2],
                        frame_number=row[3],
                        bbox=row[4],
                        confidence=row[5],
                        image_path=row[6],
                        timestamp=row[7]
                    )
                    detections.append(detection)
                
                return detections
                
        except Exception as e:
            print(f"Error getting face detections from database: {e}")
            return []
    
    def close(self):
        """Close database connections and cleanup resources."""
        # Since we're using context managers (with statements),
        # connections are automatically closed.
        # This method is provided for interface consistency.
        pass


# Factory function
def create_database_manager(db_path: str = None) -> DatabaseManager:
    """Create and return a configured database manager."""
    return DatabaseManager(db_path)
