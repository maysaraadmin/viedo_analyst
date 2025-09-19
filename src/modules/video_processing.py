"""
Video processing module for handling video capture, playback, and frame management.
"""

import cv2
import time
import threading
from collections import deque
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from core.config import config


@dataclass
class VideoInfo:
    """Data class for video information."""
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    
    @classmethod
    def from_capture(cls, path: str, cap: cv2.VideoCapture) -> 'VideoInfo':
        """Create VideoInfo from VideoCapture object."""
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        return cls(path, width, height, fps, frame_count, duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for display."""
        return {
            'Resolution': f"{self.width}x{self.height}",
            'FPS': f"{self.fps:.2f}",
            'Duration': f"{self.duration:.2f} seconds",
            'Total Frames': str(self.frame_count)
        }


class VideoCapture:
    """Enhanced video capture with better error handling and management."""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_info: Optional[VideoInfo] = None
        self.current_frame = 0
        self.is_running = False
        self._lock = threading.Lock()
    
    def open(self, source: str) -> bool:
        """Open video file or camera source."""
        try:
            with self._lock:
                if self.cap:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(source)
                if not self.cap.isOpened():
                    return False
                
                self.video_info = VideoInfo.from_capture(source, self.cap)
                self.current_frame = 0
                return True
                
        except Exception as e:
            print(f"Error opening video: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """Read next frame with error handling."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
            return ret, frame
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def get_frame_position(self) -> int:
        """Get current frame position."""
        return self.current_frame
    
    def get_progress(self) -> float:
        """Get playback progress as percentage."""
        if not self.video_info or self.video_info.frame_count == 0:
            return 0.0
        return (self.current_frame / self.video_info.frame_count) * 100
    
    def set_frame_position(self, frame_number: int) -> bool:
        """Set frame position."""
        if not self.cap or not self.cap.isOpened():
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame = frame_number
            return True
        except Exception as e:
            print(f"Error setting frame position: {e}")
            return False
    
    def get_frame_at_position(self, frame_number: int) -> Optional[cv2.Mat]:
        """Get frame at specific position."""
        if not self.cap or not self.cap.isOpened():
            return None
        
        try:
            # Set position
            if not self.set_frame_position(frame_number):
                return None
            
            # Read frame
            ret, frame = self.read()
            if ret:
                return frame
            else:
                return None
        except Exception as e:
            print(f"Error getting frame at position {frame_number}: {e}")
            return None
    
    def release(self):
        """Release video capture resources."""
        with self._lock:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.video_info = None
            self.current_frame = 0
    
    def is_opened(self) -> bool:
        """Check if video capture is opened."""
        return self.cap is not None and self.cap.isOpened()
    
    def get_info(self) -> Optional[VideoInfo]:
        """Get video information."""
        return self.video_info


class FrameBuffer:
    """Thread-safe frame buffer for smooth video playback."""
    
    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.max_frame_queue_size
        self.buffer = deque(maxlen=self.max_size)
        self._lock = threading.Lock()
        self.frame_count = 0
    
    def add_frame(self, frame: cv2.Mat) -> bool:
        """Add frame to buffer."""
        try:
            with self._lock:
                self.buffer.append(frame)
                self.frame_count += 1
                return True
        except Exception as e:
            print(f"Error adding frame to buffer: {e}")
            return False
    
    def get_frame(self) -> Optional[cv2.Mat]:
        """Get frame from buffer."""
        with self._lock:
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    def clear(self):
        """Clear all frames from buffer."""
        with self._lock:
            self.buffer.clear()
            self.frame_count = 0
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self.buffer) == 0


class VideoProcessor:
    """Main video processing class that coordinates capture and buffering."""
    
    def __init__(self):
        self.capture = VideoCapture()
        self.frame_buffer = FrameBuffer()
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def load_video(self, source: str) -> bool:
        """Load video file."""
        if self.is_processing:
            self.stop_processing()
        
        success = self.capture.open(source)
        if success:
            self.frame_buffer.clear()
        return success
    
    def start_processing(self) -> bool:
        """Start video processing thread."""
        if self.is_processing or not self.capture.is_opened():
            return False
        
        self.is_processing = True
        self._stop_event.clear()
        
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def stop_processing(self):
        """Stop video processing thread."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        self._stop_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        self.frame_buffer.clear()
    
    def _process_frames(self):
        """Process frames in separate thread."""
        while self.is_processing and not self._stop_event.is_set():
            ret, frame = self.capture.read()
            
            if not ret:
                self.is_processing = False
                break
            
            # Add frame to buffer
            if not self.frame_buffer.add_frame(frame):
                print("Warning: Failed to add frame to buffer")
            
            # Control processing speed
            time.sleep(config.frame_delay)
    
    def get_next_frame(self) -> Optional[cv2.Mat]:
        """Get next processed frame."""
        return self.frame_buffer.get_frame()
    
    def get_frame(self, frame_position: int) -> Optional[cv2.Mat]:
        """Get frame at specific position."""
        return self.capture.get_frame_at_position(frame_position)
    
    def get_video_info(self) -> Optional[VideoInfo]:
        """Get current video information."""
        return self.capture.get_info()
    
    def get_progress(self) -> float:
        """Get current playback progress."""
        return self.capture.get_progress()
    
    def release(self):
        """Release all resources."""
        self.stop_processing()
        self.capture.release()
    
    def is_loaded(self) -> bool:
        """Check if video is loaded."""
        return self.capture.is_opened()
    
    def is_processing_active(self) -> bool:
        """Check if processing is active."""
        return self.is_processing


# Utility functions
def get_supported_formats() -> list:
    """Get list of supported video formats."""
    return config.supported_video_formats


def validate_video_file(file_path: str) -> bool:
    """Validate if file is a supported video format."""
    if not file_path:
        return False
    
    file_ext = file_path.lower().split('.')[-1]
    supported_extensions = [fmt.replace('*.', '') for fmt in config.supported_video_formats]
    
    return file_ext in supported_extensions


def get_video_info_string(video_info: VideoInfo) -> str:
    """Get formatted video information string."""
    info_dict = video_info.to_dict()
    return '\n'.join([f"{key}: {value}" for key, value in info_dict.items()])
