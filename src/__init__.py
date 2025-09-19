"""
Video Analysis Application
A modular video analysis tool with object detection, tracking, and database capabilities.
"""

__version__ = "1.0.0"
__author__ = "Video Analysis Team"
__email__ = "support@videoanalysis.com"

# Import main components for easy access
from .config import config
from .video_processing import VideoProcessor, VideoInfo
from .object_detection import YOLODetector, DetectionResult
from .tracking import PersonTracker, TrackedObject
from .database import DatabaseManager, AnalysisSession
from .ui import VideoAnalysisGUI

__all__ = [
    'config',
    'VideoProcessor',
    'VideoInfo', 
    'YOLODetector',
    'DetectionResult',
    'PersonTracker',
    'TrackedObject',
    'DatabaseManager',
    'AnalysisSession',
    'VideoAnalysisGUI'
]
