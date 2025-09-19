"""
Video Storage module for handling video file storage and management operations.
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional, List
from datetime import datetime
import cv2

from core.config import config
from modules.database import StoredVideo, DatabaseManager


class VideoStorageManager:
    """Manager for video storage operations."""
    
    def __init__(self, db_manager: DatabaseManager = None):
        self.db_manager = db_manager or DatabaseManager()
        self.storage_directory = config.video_storage_directory
        
    def store_video(self, source_path: str) -> Optional[StoredVideo]:
        """Store a video file from source path to storage directory."""
        try:
            # Validate source file
            if not os.path.exists(source_path):
                print(f"Source file does not exist: {source_path}")
                return None
            
            # Get video information
            video_info = self._get_video_info(source_path)
            if not video_info:
                print(f"Failed to get video info for: {source_path}")
                return None
            
            # Generate unique filename
            filename = self._generate_unique_filename(source_path)
            stored_path = self.storage_directory / filename
            
            # Copy video file
            shutil.copy2(source_path, stored_path)
            
            # Create stored video record
            stored_video = StoredVideo(
                original_path=source_path,
                stored_path=str(stored_path),
                filename=filename,
                file_size=os.path.getsize(stored_path),
                duration=video_info['duration'],
                width=video_info['width'],
                height=video_info['height'],
                fps=video_info['fps'],
                frame_count=video_info['frame_count'],
                storage_date=datetime.now().isoformat(),
                last_accessed=datetime.now().isoformat()
            )
            
            # Save to database
            video_id = self.db_manager.save_stored_video(stored_video)
            if video_id:
                stored_video.video_id = video_id
                return stored_video
            else:
                # If database save fails, delete the copied file
                if os.path.exists(stored_path):
                    os.remove(stored_path)
                return None
                
        except Exception as e:
            print(f"Error storing video: {e}")
            return None
    
    def get_stored_video(self, video_id: int) -> Optional[StoredVideo]:
        """Get stored video by ID."""
        return self.db_manager.get_stored_video(video_id)
    
    def get_all_stored_videos(self) -> List[StoredVideo]:
        """Get all stored videos."""
        return self.db_manager.get_all_stored_videos()
    
    def delete_stored_video(self, video_id: int) -> bool:
        """Delete stored video file and database record."""
        try:
            # Get video info
            video = self.db_manager.get_stored_video(video_id)
            if not video:
                return False
            
            # Delete file
            if os.path.exists(video.stored_path):
                os.remove(video.stored_path)
            
            # Delete database record
            return self.db_manager.delete_stored_video(video_id)
            
        except Exception as e:
            print(f"Error deleting stored video: {e}")
            return False
    
    def update_access_time(self, video_id: int) -> bool:
        """Update last accessed time for a video."""
        return self.db_manager.update_video_access_time(video_id)
    
    def get_video_path(self, video_id: int) -> Optional[str]:
        """Get the file path for a stored video."""
        video = self.db_manager.get_stored_video(video_id)
        return video.stored_path if video else None
    
    def get_storage_info(self) -> dict:
        """Get storage information and statistics."""
        try:
            videos = self.get_all_stored_videos()
            
            total_files = len(videos)
            total_size = sum(v.file_size for v in videos)
            total_duration = sum(v.duration for v in videos)
            
            # Get directory size
            directory_size = 0
            if os.path.exists(self.storage_directory):
                for root, dirs, files in os.walk(self.storage_directory):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            directory_size += os.path.getsize(file_path)
            
            return {
                'total_videos': total_files,
                'total_size_bytes': total_size,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'total_size_gb': round(total_size / (1024 * 1024 * 1024), 2),
                'total_duration_seconds': total_duration,
                'total_duration_formatted': self._format_duration(total_duration),
                'directory_size_bytes': directory_size,
                'directory_size_mb': round(directory_size / (1024 * 1024), 2),
                'storage_directory': str(self.storage_directory)
            }
            
        except Exception as e:
            print(f"Error getting storage info: {e}")
            return {}
    
    def cleanup_orphaned_files(self) -> int:
        """Remove files that exist in storage but not in database."""
        try:
            # Get all stored videos from database
            stored_videos = self.get_all_stored_videos()
            stored_paths = {v.stored_path for v in stored_videos}
            
            # Get all files in storage directory
            storage_files = set()
            if os.path.exists(self.storage_directory):
                for file in os.listdir(self.storage_directory):
                    file_path = os.path.join(self.storage_directory, file)
                    if os.path.isfile(file_path):
                        storage_files.add(file_path)
            
            # Find orphaned files (in storage but not in database)
            orphaned_files = storage_files - stored_paths
            
            # Delete orphaned files
            deleted_count = 0
            for file_path in orphaned_files:
                try:
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted orphaned file: {file_path}")
                except Exception as e:
                    print(f"Error deleting orphaned file {file_path}: {e}")
            
            return deleted_count
            
        except Exception as e:
            print(f"Error cleaning up orphaned files: {e}")
            return 0
    
    def _get_video_info(self, video_path: str) -> Optional[dict]:
        """Extract video information using OpenCV."""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate duration
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration
            }
            
        except Exception as e:
            print(f"Error getting video info: {e}")
            return None
    
    def _generate_unique_filename(self, source_path: str) -> str:
        """Generate a unique filename for stored video."""
        # Get original filename
        original_filename = Path(source_path).stem
        extension = Path(source_path).suffix
        
        # Add timestamp and hash for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a short hash of the source path
        path_hash = hashlib.md5(source_path.encode()).hexdigest()[:8]
        
        # Combine to create unique filename
        unique_filename = f"{original_filename}_{timestamp}_{path_hash}{extension}"
        
        return unique_filename
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in seconds to readable format."""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours}h {minutes}m {secs}s"


# Factory function
def create_video_storage_manager(db_manager: DatabaseManager = None) -> VideoStorageManager:
    """Create and return a configured video storage manager."""
    return VideoStorageManager(db_manager)
