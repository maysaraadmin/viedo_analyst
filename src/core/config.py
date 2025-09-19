"""
Configuration module for video analysis application.
Contains all settings, constants, and configuration parameters.
"""

import os
import cv2
from pathlib import Path

# Application Settings
APP_NAME = "Video Analysis Tool"
APP_VERSION = "1.0.0"
WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION}"
WINDOW_GEOMETRY = (100, 100, 1200, 800)

# Video Processing Settings
DEFAULT_FPS = 30
FRAME_DELAY = 0.033  # ~30 FPS
MAX_FRAME_QUEUE_SIZE = 5

# Object Detection Settings
DEFAULT_MODEL = 'yolov8n.pt'
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD_RANGE = (1, 99)
TRACKING_DISTANCE_THRESHOLD = 50  # pixels
TRACKING_TIMEOUT = 5.0  # seconds

# Database Settings
DEFAULT_DB_NAME = 'video_analysis.db'
DB_BACKUP_DIR = 'backups'

# UI Settings
VIDEO_LABEL_MIN_SIZE = (800, 600)
VIDEO_LABEL_BG_COLOR = "black"
OBJECT_COUNT_MAX_HEIGHT = 200
VIDEO_INFO_MAX_HEIGHT = 150

# Detection Colors
PERSON_COLOR = (0, 255, 0)  # Green
OTHER_OBJECT_COLOR = (255, 0, 0)  # Red
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2

# Supported Video Formats
SUPPORTED_VIDEO_FORMATS = [
    '*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', 
    '*.wmv', '*.mpeg', '*.mpg', '*.m4v'
]

# File Paths
def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent

def get_database_path():
    """Get the database file path."""
    return get_project_root() / DEFAULT_DB_NAME

def get_backup_directory():
    """Get the backup directory path."""
    backup_dir = get_project_root() / DB_BACKUP_DIR
    backup_dir.mkdir(exist_ok=True)
    return backup_dir

def get_video_storage_directory():
    """Get the video storage directory path."""
    storage_dir = get_project_root() / VIDEO_STORAGE_DIRECTORY
    storage_dir.mkdir(exist_ok=True)
    return storage_dir

# Logging Settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = "logs/video_analysis.log"

# Performance Settings
MAX_DETECTION_FPS = 30
MEMORY_CLEANUP_INTERVAL = 100  # frames

# File Paths
OUTPUT_DIRECTORY = "output"
TEMP_DIRECTORY = "temp"
LOGS_DIRECTORY = "logs"
VIDEO_STORAGE_DIRECTORY = "stored_videos"

# Error Messages
ERROR_MESSAGES = {
    'video_load_failed': "Failed to load video file",
    'detection_error': "Error during object detection",
    'tracking_error': "Error during person tracking",
    'database_error': "Database operation failed",
    'ui_error': "UI update failed"
}

# Success Messages
SUCCESS_MESSAGES = {
    'video_loaded': "Video loaded successfully",
    'analysis_complete': "Analysis completed",
    'data_saved': "Data saved successfully"
}

class Config:
    """Configuration class for easy access to settings."""
    
    def __init__(self):
        self.app_name = APP_NAME
        self.app_version = APP_VERSION
        self.APP_NAME = APP_NAME
        self.APP_VERSION = APP_VERSION
        self.window_title = WINDOW_TITLE
        self.window_geometry = WINDOW_GEOMETRY
        
        # Video processing
        self.default_fps = DEFAULT_FPS
        self.frame_delay = FRAME_DELAY
        self.max_frame_queue_size = MAX_FRAME_QUEUE_SIZE
        
        # Object detection
        self.default_model = DEFAULT_MODEL
        self.default_confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.default_confidence = DEFAULT_CONFIDENCE_THRESHOLD  # Alias for UI compatibility
        self.default_iou_threshold = 0.5  # Default IoU threshold for NMS
        self.confidence_threshold_range = CONFIDENCE_THRESHOLD_RANGE
        self.tracking_distance_threshold = TRACKING_DISTANCE_THRESHOLD
        self.tracking_timeout = TRACKING_TIMEOUT
        self.max_disappeared_frames = 30  # Maximum frames a person can be missing before being considered lost
        self.max_tracking_distance = 100  # Maximum distance for tracking association
        self.use_csrt_tracker = False  # Whether to use CSRT tracker for more accuracy
        self.max_trajectory_length = 50  # Maximum trajectory points to store
        
        # Database
        self.default_db_name = DEFAULT_DB_NAME
        self.db_path = get_database_path()
        self.backup_dir = get_backup_directory()
        
        # Video storage
        self.video_storage_directory = get_video_storage_directory()
        
        # UI
        self.video_label_min_size = VIDEO_LABEL_MIN_SIZE
        self.video_label_bg_color = VIDEO_LABEL_BG_COLOR
        self.object_count_max_height = OBJECT_COUNT_MAX_HEIGHT
        self.video_info_max_height = VIDEO_INFO_MAX_HEIGHT
        
        # Colors
        self.person_color = PERSON_COLOR
        self.other_object_color = OTHER_OBJECT_COLOR
        self.text_font = TEXT_FONT
        self.text_scale = TEXT_SCALE
        self.text_thickness = TEXT_THICKNESS
        
        # File formats
        self.supported_video_formats = SUPPORTED_VIDEO_FORMATS
        
        # Performance
        self.max_detection_fps = MAX_DETECTION_FPS
        self.memory_cleanup_interval = MEMORY_CLEANUP_INTERVAL
        
        # Logging
        self.log_level = LOG_LEVEL
        self.log_format = LOG_FORMAT
        self.log_file_path = get_project_root() / LOG_FILE
        
        # Directories
        self.output_directory = get_project_root() / OUTPUT_DIRECTORY
        self.temp_directory = get_project_root() / TEMP_DIRECTORY
        self.logs_directory = get_project_root() / LOGS_DIRECTORY
        
        # Messages
        self.error_messages = ERROR_MESSAGES
        self.success_messages = SUCCESS_MESSAGES

# Global config instance
config = Config()
