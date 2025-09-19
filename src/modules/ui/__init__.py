"""
UI module for GUI components and user interface management.
"""

import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QSlider, QComboBox, QFileDialog,
                             QMessageBox, QProgressBar, QSplitter, QFrame, QTextEdit,
                             QListWidget, QDialog, QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
import json
from datetime import datetime

# Import from core and other modules
from core.config import config
from modules.object_detection import create_detector
from modules.tracking import create_person_tracker
from modules.database import create_database_manager, StoredVideo, PersonProfile, FaceDetection
from modules.video_storage import create_video_storage_manager
from modules.video_processing import VideoProcessor, VideoInfo
from modules.face_detection import FaceDetector, PersonProfileManager
from modules.audio_manager import AudioManager


class VideoLibraryWidget(QWidget):
    """Widget for displaying stored videos with their information."""
    
    load_video_clicked = pyqtSignal(int)  # video_id
    delete_video_clicked = pyqtSignal(int)  # video_id
    refresh_clicked = pyqtSignal()
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.init_ui()
        self.refresh_videos()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Video Library")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.on_refresh_clicked)
        button_layout.addWidget(self.refresh_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Videos table
        self.videos_table = QTableWidget()
        self.videos_table.setColumnCount(5)
        self.videos_table.setHorizontalHeaderLabels(["Name", "Duration", "Size", "Storage Date", "Actions"])
        self.videos_table.horizontalHeader().setStretchLastSection(True)
        self.videos_table.setAlternatingRowColors(True)
        self.videos_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.videos_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Set column widths
        self.videos_table.setColumnWidth(0, 200)  # Name
        self.videos_table.setColumnWidth(1, 100)  # Duration
        self.videos_table.setColumnWidth(2, 100)  # Size
        self.videos_table.setColumnWidth(3, 150)  # Storage Date
        
        layout.addWidget(self.videos_table)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def refresh_videos(self):
        """Refresh the videos list from database."""
        try:
            videos = self.db_manager.get_all_stored_videos()
            self.update_videos_table(videos)
            self.status_label.setText(f"Loaded {len(videos)} videos")
        except Exception as e:
            self.status_label.setText(f"Error loading videos: {str(e)}")
    
    def update_videos_table(self, videos):
        """Update the videos table with stored video data."""
        self.videos_table.setRowCount(len(videos))
        
        for row, video in enumerate(videos):
            # Name column
            name_item = QTableWidgetItem(video.filename)
            name_item.setToolTip(f"Original path: {video.original_path}")
            self.videos_table.setItem(row, 0, name_item)
            
            # Duration column
            duration_text = self.format_duration(video.duration)
            duration_item = QTableWidgetItem(duration_text)
            self.videos_table.setItem(row, 1, duration_item)
            
            # Size column
            size_text = self.format_file_size(video.file_size)
            size_item = QTableWidgetItem(size_text)
            self.videos_table.setItem(row, 2, size_item)
            
            # Storage date column
            storage_date = video.storage_date.split('T')[0] if 'T' in video.storage_date else video.storage_date
            date_item = QTableWidgetItem(storage_date)
            self.videos_table.setItem(row, 3, date_item)
            
            # Actions column
            actions_widget = QWidget()
            actions_layout = QHBoxLayout()
            actions_layout.setContentsMargins(2, 2, 2, 2)
            
            load_button = QPushButton("Load")
            load_button.setMaximumWidth(60)
            load_button.clicked.connect(lambda checked, vid=video.video_id: self.on_load_video_clicked(vid))
            actions_layout.addWidget(load_button)
            
            delete_button = QPushButton("Delete")
            delete_button.setMaximumWidth(60)
            delete_button.clicked.connect(lambda checked, vid=video.video_id: self.on_delete_video_clicked(vid))
            actions_layout.addWidget(delete_button)
            
            actions_widget.setLayout(actions_layout)
            self.videos_table.setCellWidget(row, 4, actions_widget)
    
    def format_duration(self, duration_seconds):
        """Format duration in seconds to HH:MM:SS format."""
        if duration_seconds <= 0:
            return "00:00:00"
        
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def format_file_size(self, size_bytes):
        """Format file size in bytes to human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024 and i < len(size_names) - 1:
            size /= 1024
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def on_refresh_clicked(self):
        """Handle refresh button click."""
        self.refresh_clicked.emit()
        self.refresh_videos()
    
    def on_load_video_clicked(self, video_id):
        """Handle load video button click."""
        self.load_video_clicked.emit(video_id)
    
    def on_delete_video_clicked(self, video_id):
        """Handle delete video button click."""
        reply = QMessageBox.question(
            self, 
            'Delete Video', 
            'Are you sure you want to delete this video from the library?',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_video_clicked.emit(video_id)


class PersonProfileWidget(QWidget):
    """Widget for displaying and managing person profiles."""
    
    edit_profile_clicked = pyqtSignal(int)  # person_id
    delete_profile_clicked = pyqtSignal(int)  # person_id
    refresh_clicked = pyqtSignal()
    
    def __init__(self, db_manager):
        super().__init__()
        self.db_manager = db_manager
        self.face_detector = FaceDetector()
        self.profile_manager = PersonProfileManager(self.db_manager)
        self.init_ui()
        self.refresh_profiles()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Title
        title_label = QLabel("Person Profiles")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.on_refresh_clicked)
        button_layout.addWidget(self.refresh_button)
        
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        
        # Profiles table
        self.profiles_table = QTableWidget()
        self.profiles_table.setColumnCount(6)
        self.profiles_table.setHorizontalHeaderLabels(["Name", "First Seen", "Last Seen", "Appearances", "Face Images", "Actions"])
        self.profiles_table.horizontalHeader().setStretchLastSection(True)
        self.profiles_table.setAlternatingRowColors(True)
        self.profiles_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.profiles_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # Set column widths
        self.profiles_table.setColumnWidth(0, 150)  # Name
        self.profiles_table.setColumnWidth(1, 120)  # First Seen
        self.profiles_table.setColumnWidth(2, 120)  # Last Seen
        self.profiles_table.setColumnWidth(3, 80)   # Appearances
        self.profiles_table.setColumnWidth(4, 80)   # Face Images
        
        layout.addWidget(self.profiles_table)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
    
    def refresh_profiles(self):
        """Refresh the profiles list from database."""
        try:
            profiles = self.db_manager.get_all_person_profiles()
            self.update_profiles_table(profiles)
            self.status_label.setText(f"Loaded {len(profiles)} profiles")
        except Exception as e:
            self.status_label.setText(f"Error loading profiles: {str(e)}")
    
    def update_profiles_table(self, profiles):
        """Update the profiles table with person profile data."""
        self.profiles_table.setRowCount(len(profiles))
        
        for row, profile in enumerate(profiles):
            # Name column
            name_item = QTableWidgetItem(profile.profile_name)
            self.profiles_table.setItem(row, 0, name_item)
            
            # First Seen column
            first_seen = self.format_datetime(profile.first_seen)
            first_seen_item = QTableWidgetItem(first_seen)
            self.profiles_table.setItem(row, 1, first_seen_item)
            
            # Last Seen column
            last_seen = self.format_datetime(profile.last_seen)
            last_seen_item = QTableWidgetItem(last_seen)
            self.profiles_table.setItem(row, 2, last_seen_item)
            
            # Appearances column
            appearances_item = QTableWidgetItem(str(profile.total_appearances))
            self.profiles_table.setItem(row, 3, appearances_item)
            
            # Face Images column
            face_images_count = len(json.loads(profile.face_images)) if profile.face_images else 0
            face_images_item = QTableWidgetItem(str(face_images_count))
            self.profiles_table.setItem(row, 4, face_images_item)
            
            # Actions column
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(0, 0, 0, 0)
            
            # Edit button
            edit_btn = QPushButton("Edit")
            edit_btn.setMaximumWidth(60)
            edit_btn.clicked.connect(lambda checked, pid=profile.person_id: self.on_edit_profile_clicked(pid))
            actions_layout.addWidget(edit_btn)
            
            # Delete button
            delete_btn = QPushButton("Delete")
            delete_btn.setMaximumWidth(60)
            delete_btn.clicked.connect(lambda checked, pid=profile.person_id: self.on_delete_profile_clicked(pid))
            actions_layout.addWidget(delete_btn)
            
            actions_layout.addStretch()
            self.profiles_table.setCellWidget(row, 5, actions_widget)
    
    def format_datetime(self, datetime_str):
        """Format datetime string for display."""
        try:
            dt = datetime.fromisoformat(datetime_str)
            return dt.strftime("%Y-%m-%d %H:%M")
        except:
            return datetime_str
    
    def on_refresh_clicked(self):
        """Handle refresh button click."""
        self.refresh_clicked.emit()
        self.refresh_profiles()
    
    def on_edit_profile_clicked(self, person_id):
        """Handle edit profile button click."""
        self.edit_profile_clicked.emit(person_id)
    
    def on_delete_profile_clicked(self, person_id):
        """Handle delete profile button click."""
        reply = QMessageBox.question(
            self, 
            'Delete Profile', 
            'Are you sure you want to delete this person profile?\nThis will also delete all associated face detections.',
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_profile_clicked.emit(person_id)


class VideoAnalysisGUI(QMainWindow):
    """Main GUI class for video analysis application."""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.video_processor = None
        self.detector = None
        self.tracker = None
        self.db_manager = None
        self.video_storage_manager = None
        self.current_video_path = None
        self.current_session_id = None
        self.video_thread = None
        self.is_playing = False
        self.current_frame = None
        self.detection_results = []
        self.tracking_results = []
        self.detection_enabled = False
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        
        # Initialize database manager
        self.db_manager = create_database_manager()
        
        # Initialize audio manager
        self.audio_manager = AudioManager()
        
        self.setup_ui()
        self.setup_connections()
        
        # Initialize data lists
        self.refresh_sessions()
        
    def setup_ui(self):
        """Setup the user interface."""
        self.setWindowTitle(config.window_title)
        self.setGeometry(*config.window_geometry)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top toolbar
        toolbar_layout = QHBoxLayout()
        
        # Load video button
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar_layout.addWidget(self.load_video_btn)
        
        # Store video button
        self.store_video_btn = QPushButton("Store Video")
        self.store_video_btn.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar_layout.addWidget(self.store_video_btn)
        
        # Manage stored videos button
        self.manage_videos_btn = QPushButton("Manage Videos")
        self.manage_videos_btn.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar_layout.addWidget(self.manage_videos_btn)
        
        # Play/Pause + Detection button
        self.play_pause_btn = QPushButton("Play + Detection")
        self.play_pause_btn.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar_layout.addWidget(self.play_pause_btn)
        
        
        # Save session button
        self.save_session_btn = QPushButton("Save Session")
        self.save_session_btn.setFont(QFont("Arial", 10, QFont.Bold))
        toolbar_layout.addWidget(self.save_session_btn)
        
        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)
        
        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)
        
        # Video display area
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 2px solid #cccccc; background-color: #000000;")
        video_layout.addWidget(self.video_label)
        
        # Video controls
        controls_layout = QHBoxLayout()
        
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setMinimum(0)
        self.position_slider.setMaximum(100)
        self.position_slider.setValue(0)
        controls_layout.addWidget(self.position_slider)
        
        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setFont(QFont("Arial", 9))
        controls_layout.addWidget(self.time_label)
        
        # Audio controls
        audio_layout = QHBoxLayout()
        
        # Mute button
        self.mute_btn = QPushButton("🔊")
        self.mute_btn.setFont(QFont("Arial", 10))
        self.mute_btn.setMaximumWidth(40)
        audio_layout.addWidget(self.mute_btn)
        
        # Volume slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        if self.audio_manager:
            self.volume_slider.setValue(int(self.audio_manager.volume * 100))
        self.volume_slider.setMaximumWidth(100)
        audio_layout.addWidget(self.volume_slider)
        
        # Volume label
        self.volume_label = QLabel("70%")
        self.volume_label.setFont(QFont("Arial", 9))
        audio_layout.addWidget(self.volume_label)
        
        audio_layout.addStretch()
        controls_layout.addLayout(audio_layout)
        
        video_layout.addLayout(controls_layout)
        content_splitter.addWidget(video_widget)
        
        # Right panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Detection settings
        settings_group = QFrame()
        settings_group.setFrameShape(QFrame.StyledPanel)
        settings_layout = QVBoxLayout(settings_group)
        
        settings_label = QLabel("Detection Settings")
        settings_label.setFont(QFont("Arial", 12, QFont.Bold))
        settings_layout.addWidget(settings_label)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.setCurrentText(config.default_model)
        model_layout.addWidget(self.model_combo)
        settings_layout.addLayout(model_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_label = QLabel("Confidence:")
        conf_layout.addWidget(conf_label)
        
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(0)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(int(config.default_confidence * 100))
        self.conf_label = QLabel(f"{config.default_confidence:.2f}")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)
        
        # IOU threshold
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IOU Threshold:")
        iou_layout.addWidget(iou_label)
        
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setMinimum(0)
        self.iou_slider.setMaximum(100)
        self.iou_slider.setValue(int(config.default_iou_threshold * 100))
        self.iou_label = QLabel(f"{config.default_iou_threshold:.2f}")
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_label)
        settings_layout.addLayout(iou_layout)
        
        right_layout.addWidget(settings_group)
        
        # Detection results
        results_group = QFrame()
        results_group.setFrameShape(QFrame.StyledPanel)
        results_layout = QVBoxLayout(results_group)
        
        results_label = QLabel("Detection Results")
        results_label.setFont(QFont("Arial", 12, QFont.Bold))
        results_layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        right_layout.addWidget(results_group)
        
        # Tab widget for Library and Profiles
        tab_widget = QTabWidget()
        
        # Video Library tab
        self.video_library = VideoLibraryWidget(self.db_manager)
        tab_widget.addTab(self.video_library, "Video Library")
        
        # Person Profiles tab
        self.person_profiles = PersonProfileWidget(self.db_manager)
        tab_widget.addTab(self.person_profiles, "Person Profiles")
        
        right_layout.addWidget(tab_widget)
        
        # Session management
        session_group = QFrame()
        session_group.setFrameShape(QFrame.StyledPanel)
        session_layout = QVBoxLayout(session_group)
        
        session_label = QLabel("Session Management")
        session_label.setFont(QFont("Arial", 12, QFont.Bold))
        session_layout.addWidget(session_label)
        
        self.session_combo = QComboBox()
        self.session_combo.addItem("Select Session...")
        session_layout.addWidget(self.session_combo)
        
        session_buttons_layout = QHBoxLayout()
        self.load_session_btn = QPushButton("Load Session")
        self.delete_session_btn = QPushButton("Delete Session")
        session_buttons_layout.addWidget(self.load_session_btn)
        session_buttons_layout.addWidget(self.delete_session_btn)
        session_layout.addLayout(session_buttons_layout)
        
        right_layout.addWidget(session_group)
        
        right_layout.addStretch()
        content_splitter.addWidget(right_panel)
        
        # Set splitter sizes
        content_splitter.setSizes([700, 300])
        main_layout.addWidget(content_splitter)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def setup_connections(self):
        """Setup signal connections."""
        self.load_video_btn.clicked.connect(self.load_video)
        self.store_video_btn.clicked.connect(self.store_video)
        self.manage_videos_btn.clicked.connect(self.manage_stored_videos)
        self.play_pause_btn.clicked.connect(self.toggle_play_pause_with_detection)
        self.save_session_btn.clicked.connect(self.save_session)
        self.load_session_btn.clicked.connect(self.load_session)
        # Video library connections
        self.video_library.load_video_clicked.connect(self.load_stored_video)
        self.video_library.delete_video_clicked.connect(self.delete_stored_video)
        self.video_library.refresh_clicked.connect(self.refresh_video_library)
        
        # Person profiles connections
        self.person_profiles.edit_profile_clicked.connect(self.edit_person_profile)
        self.person_profiles.delete_profile_clicked.connect(self.delete_person_profile)
        self.person_profiles.refresh_clicked.connect(self.refresh_person_profiles)
        
        self.delete_session_btn.clicked.connect(self.delete_session)
        
        self.position_slider.sliderPressed.connect(self.on_slider_pressed)
        self.position_slider.sliderReleased.connect(self.on_slider_released)
        
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)
        self.iou_slider.valueChanged.connect(self.on_iou_changed)
        
        # Audio control connections
        self.mute_btn.clicked.connect(self.toggle_mute)
        self.volume_slider.valueChanged.connect(self.on_volume_changed)
        
        self.session_combo.currentIndexChanged.connect(self.on_session_selected)
        
    def load_video(self):
        """Load a video file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)"
        )
        
        if file_path:
            self.current_video_path = file_path
            self.status_bar.showMessage(f"Loading video: {file_path}")
            
            # Initialize video processor
            if self.video_processor:
                self.video_processor.release()
            
            self.video_processor = VideoProcessor()
            if self.video_processor.load_video(file_path):
                video_info = self.video_processor.get_video_info()
                self.position_slider.setMaximum(video_info.frame_count - 1)
                self.update_time_display(0, video_info.frame_count)
                self.status_bar.showMessage(f"Video loaded: {video_info.width}x{video_info.height}, {video_info.fps:.2f} fps, {video_info.frame_count} frames")
                
                # Load first frame
                frame = self.video_processor.get_frame(0)
                if frame is not None:
                    self.display_frame(frame)
                    
                # Start new session
                self.start_new_session()
            else:
                self.status_bar.showMessage("Failed to load video")
    
    def start_new_session(self):
        """Start a new session by clearing current session data."""
        self.current_session_id = None
        # Reset any session-specific UI elements if they exist
        if hasattr(self, 'session_combo') and self.session_combo:
            self.session_combo.setCurrentIndex(0)
        self.status_bar.showMessage("New session started")
            
    def toggle_play_pause(self):
        """Toggle between play and pause."""
        if not self.current_video_path:
            self.status_bar.showMessage("No video loaded")
            return
            
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()
            
    def toggle_play_pause_with_detection(self):
        """Toggle between play and pause with automatic detection."""
        if not self.current_video_path:
            self.status_bar.showMessage("No video loaded")
            return
            
        if self.is_playing:
            # Check if video is at the end
            if self.position_slider.value() >= self.position_slider.maximum() - 1:
                # If at the end, stop completely (like stop button)
                self.stop_video()
            else:
                # If not at end, just pause and stop detection
                self.pause_video()
                if self.detection_enabled:
                    self.stop_detection()
                # Reset button to initial state
                self.play_pause_btn.setText("Play + Detection")
        else:
            # If not playing, start video and enable detection
            self.play_video()
            if not self.detection_enabled:
                self.start_detection()
            # Update button to show detection is active
            self.play_pause_btn.setText("Pause (Detecting)")
            
    def play_video(self):
        """Start video playback."""
        if not self.video_thread or not self.video_thread.isRunning():
            self.video_thread = VideoThread(self.video_processor, self.db_manager)
            
            # Always set detector and tracker if available (not None)
            if hasattr(self, 'detector') and self.detector is not None:
                self.video_thread.set_detector(self.detector)
                print(f"DEBUG: Set detector in video thread: {self.detector is not None}")
            else:
                print(f"DEBUG: No detector available in GUI")
                
            if hasattr(self, 'tracker') and self.tracker is not None:
                self.video_thread.set_tracker(self.tracker)
                print(f"DEBUG: Set tracker in video thread: {self.tracker is not None}")
            else:
                print(f"DEBUG: No tracker available in GUI")
                
            # Set detection settings
            self.video_thread.set_detection_enabled(self.detection_enabled)
            self.video_thread.set_confidence_threshold(self.confidence_threshold)
            self.video_thread.set_iou_threshold(self.iou_threshold)
            
            # Connect signals
            self.video_thread.frame_ready.connect(self.display_frame)
            self.video_thread.position_changed.connect(self.update_position)
            self.video_thread.finished.connect(self.on_video_finished)
            self.video_thread.start()
            
            # Start audio playback synchronized with video
            self.start_audio_playback()
            
        self.is_playing = True
        self.play_pause_btn.setText("Pause")
        self.status_bar.showMessage("Playing video with detection")
        
    def pause_video(self):
        """Pause video playback."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause()
            
        # Pause audio playback
        self.stop_audio_playback()
            
        self.is_playing = False
        self.play_pause_btn.setText("Play + Detection")
        self.status_bar.showMessage("Video paused")
        
    def stop_video(self):
        """Stop video playback."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
            
        # Stop audio playback
        self.stop_audio_playback()
            
        self.is_playing = False
        self.play_pause_btn.setText("Play + Detection")
        self.position_slider.setValue(0)
        
        # Stop detection if it's enabled
        if self.detection_enabled:
            self.stop_detection()
        
        if self.video_processor:
            frame = self.video_processor.get_frame(0)
            if frame is not None:
                self.display_frame(frame)
                
        self.status_bar.showMessage("Video stopped")
        
    def toggle_detection(self):
        """Toggle object detection."""
        print(f"DEBUG: toggle_detection called, current_video_path: {self.current_video_path}")
        
        if not self.current_video_path:
            self.status_bar.showMessage("No video loaded")
            print("DEBUG: No video loaded, returning")
            return
            
        if not self.detection_enabled:
            print("DEBUG: Calling start_detection")
            self.start_detection()
        else:
            print("DEBUG: Calling stop_detection")
            self.stop_detection()
            
    def start_detection(self):
        """Start object detection."""
        print("DEBUG: start_detection method called")
        try:
            if not self.detector:
                print("DEBUG: Creating new detector")
                # Check if model_combo still exists
                if hasattr(self, 'model_combo') and self.model_combo:
                    model_name = self.model_combo.currentText()
                else:
                    model_name = config.default_model
                self.detector = create_detector(model_name)
                print(f"DEBUG: Detector created: {self.detector is not None}")
            else:
                print("DEBUG: Using existing detector")
                
            if not self.tracker:
                print("DEBUG: Creating new tracker")
                self.tracker = create_person_tracker()
                print(f"DEBUG: Tracker created: {self.tracker is not None}")
            else:
                print("DEBUG: Using existing tracker")
                
        except Exception as e:
            print(f"Error starting detection: {e}")
            return
            
        # Set detection enabled flag
        self.detection_enabled = True
        print(f"DEBUG: Set detection_enabled to True")
        
        # Update play button text to indicate detection is active
        if self.is_playing:
            self.play_pause_btn.setText("Pause (Detecting)")
        self.status_bar.showMessage("Detection started")
        
        # Enable detection in video thread
        if self.video_thread:
            print("DEBUG: Calling enable_detection on video_thread")
            self.video_thread.enable_detection(
                self.detector, 
                self.tracker, 
                self.conf_slider.value() / 100.0,
                self.iou_slider.value() / 100.0
            )
        else:
            print("DEBUG: No video_thread available")
            
    def stop_detection(self):
        """Stop object detection."""
        # Set detection enabled flag
        self.detection_enabled = False
        
        # Update play button text to indicate detection is stopped
        if self.is_playing:
            self.play_pause_btn.setText("Pause")
        self.status_bar.showMessage("Detection stopped")
        
        # Disable detection in video thread
        if self.video_thread:
            self.video_thread.disable_detection()
            
    def display_frame(self, frame):
        """Display a frame in the video label."""
        self.current_frame = frame.copy()
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap and display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_position(self, position):
        """Update the current position."""
        self.position_slider.setValue(position)
        if self.video_processor:
            total_frames = self.video_processor.get_video_info().frame_count
            self.update_time_display(position, total_frames)
            
    def update_time_display(self, current_frame, total_frames):
        """Update the time display."""
        if self.video_processor:
            fps = self.video_processor.get_video_info().fps
            current_time = current_frame / fps
            total_time = total_frames / fps
            
            current_str = f"{int(current_time//3600):02d}:{int((current_time%3600)//60):02d}:{int(current_time%60):02d}"
            total_str = f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}"
            
            self.time_label.setText(f"{current_str} / {total_str}")
            
    def on_slider_pressed(self):
        """Handle slider press."""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.pause()
            
    def on_slider_released(self):
        """Handle slider release."""
        if self.video_processor:
            frame = self.video_processor.get_frame(self.position_slider.value())
            if frame is not None:
                self.display_frame(frame)
                
        if self.is_playing and self.video_thread:
            self.video_thread.set_position(self.position_slider.value())
            self.video_thread.resume()
            
    def on_confidence_changed(self, value):
        """Handle confidence threshold change."""
        confidence = value / 100.0
        self.conf_label.setText(f"{confidence:.2f}")
        
        if self.video_thread:
            self.video_thread.set_confidence_threshold(confidence)
            
    def on_iou_changed(self, value):
        """Handle IOU threshold change."""
        iou_threshold = value / 100.0
        self.iou_label.setText(f"{iou_threshold:.2f}")
        
        if self.video_thread:
            self.video_thread.set_iou_threshold(iou_threshold)
            
    def toggle_mute(self):
        """Toggle audio mute state."""
        if self.audio_manager and self.audio_manager.audio_available:
            muted = self.audio_manager.toggle_mute()
            if muted:
                self.mute_btn.setText("🔇")
                self.status_bar.showMessage("Audio muted")
            else:
                self.mute_btn.setText("🔊")
                self.status_bar.showMessage("Audio unmuted")
        else:
            self.status_bar.showMessage("Audio not available")
            
    def on_volume_changed(self, value):
        """Handle volume slider change."""
        volume = value / 100.0
        if self.audio_manager:
            self.audio_manager.set_volume(volume)
        self.volume_label.setText(f"{value}%")
    
    def toggle_mute(self):
        """Toggle audio mute state."""
        if self.audio_manager and self.audio_manager.audio_available:
            muted = self.audio_manager.toggle_mute()
            # Update mute button icon
            self.mute_btn.setText("🔇" if muted else "🔊")
            # Update status
            if muted:
                self.status_bar.showMessage("Audio muted")
            else:
                self.status_bar.showMessage("Audio unmuted")
        else:
            self.status_bar.showMessage("Audio not available")
        
    def start_audio_playback(self, start_time=0.0):
        """Start audio playback synchronized with video."""
        if self.audio_manager and self.audio_manager.audio_available and self.current_video_path:
            # Extract audio if not already extracted
            if not self.audio_manager.current_audio_path:
                audio_path = self.audio_manager.extract_audio_from_video(self.current_video_path)
                if audio_path:
                    self.audio_manager.play_audio(audio_path, start_time)
                    self.status_bar.showMessage("Audio playback started")
                else:
                    self.status_bar.showMessage("No audio track found in video")
            else:
                # Resume existing audio playback
                self.audio_manager.play_audio(self.audio_manager.current_audio_path, start_time)
                self.status_bar.showMessage("Audio playback resumed")
        else:
            self.status_bar.showMessage("Audio not available")
            
    def stop_audio_playback(self):
        """Stop audio playback."""
        if self.audio_manager and self.audio_manager.audio_available:
            self.audio_manager.stop_audio()
            self.status_bar.showMessage("Audio playback stopped")
            
    def on_video_finished(self):
        """Handle video playback finished."""
        self.play_pause_btn.setText("Play")
        # Stop audio playback
        self.stop_audio_playback()
        self.status_bar.showMessage("Video playback finished")
        
    def save_session(self):
        """Save the current session."""
        if not self.current_session_id:
            self.status_bar.showMessage("No active session")
            return
            
        if not self.db_manager:
            self.db_manager = create_database_manager(config.db_path)
            
        session_data = {
            'end_time': datetime.now().isoformat(),
            'detection_results': self.detection_results,
            'tracking_results': self.tracking_results
        }
        
        self.db_manager.update_session(self.current_session_id, session_data)
        self.status_bar.showMessage(f"Session saved: {self.current_session_id}")
        
    def load_session(self):
        """Load a selected session."""
        try:
            # Check if session_combo still exists
            if hasattr(self, 'session_combo') and self.session_combo:
                session_name = self.session_combo.currentText()
            else:
                self.status_bar.showMessage("Session selector not available")
                return
                
            if session_name == "Select Session...":
                return
                
            session_id = int(session_name.split(" - ")[0])
            
            if not self.db_manager:
                self.db_manager = create_database_manager(config.db_path)
                
            session_data = self.db_manager.get_session(session_id)
            if session_data:
                self.current_session_id = session_id
                self.current_video_path = session_data['video_path']
                
                # Load video
                if self.video_processor:
                    self.video_processor.release()
                    
                self.video_processor = VideoProcessor()
                if self.video_processor.load_video(self.current_video_path):
                    video_info = self.video_processor.get_video_info()
                    
                    # Check if UI components exist before using them
                    if hasattr(self, 'position_slider') and self.position_slider:
                        self.position_slider.setMaximum(video_info.frame_count - 1)
                    
                    self.update_time_display(0, video_info.frame_count)
                    
                    # Load first frame
                    frame = self.video_processor.get_frame(0)
                    if frame is not None:
                        self.display_frame(frame)
                        
                    # Load session settings with safety checks
                    if hasattr(self, 'model_combo') and self.model_combo:
                        self.model_combo.setCurrentText(session_data.get('model', config.default_model))
                    
                    if hasattr(self, 'conf_slider') and self.conf_slider:
                        self.conf_slider.setValue(int(session_data.get('confidence_threshold', config.default_confidence) * 100))
                    
                    if hasattr(self, 'iou_slider') and self.iou_slider:
                        self.iou_slider.setValue(int(session_data.get('iou_threshold', config.default_iou_threshold) * 100))
                    
                    self.status_bar.showMessage(f"Session loaded: {session_id}")
                else:
                    self.status_bar.showMessage("Failed to load video from session")
            else:
                self.status_bar.showMessage("Failed to load session")
                
        except Exception as e:
            print(f"Error loading session: {e}")
            self.status_bar.showMessage("Error loading session")
            
    def delete_session(self):
        """Delete a selected session."""
        try:
            # Check if session_combo still exists
            if hasattr(self, 'session_combo') and self.session_combo:
                session_name = self.session_combo.currentText()
            else:
                self.status_bar.showMessage("Session selector not available")
                return
                
            if session_name == "Select Session...":
                return
                
            session_id = int(session_name.split(" - ")[0])
            
            if not self.db_manager:
                self.db_manager = create_database_manager(config.db_path)
                
            if self.db_manager.delete_session(session_id):
                self.refresh_sessions()
                self.status_bar.showMessage(f"Session deleted: {session_id}")
            else:
                self.status_bar.showMessage("Failed to delete session")
                
        except Exception as e:
            print(f"Error deleting session: {e}")
            self.status_bar.showMessage("Error deleting session")
            
    def refresh_sessions(self):
        """Refresh the session list."""
        try:
            # Check if GUI is ready and visible
            if not self.isVisible():
                return
                
            if not self.db_manager:
                self.db_manager = create_database_manager(config.db_path)
                
            sessions = self.db_manager.get_all_sessions()
            
            # Check if session_combo still exists
            if hasattr(self, 'session_combo') and self.session_combo:
                self.session_combo.clear()
                self.session_combo.addItem("Select Session...")
                
                for session in sessions:
                    session_name = f"{session['id']} - {session['video_path']}"
                    self.session_combo.addItem(session_name)
            else:
                print("Session combo box not available")
                
        except Exception as e:
            print(f"Error refreshing sessions: {e}")
    
    def store_video(self):
        """Store the current video in the storage system."""
        if not self.current_video_path:
            QMessageBox.warning(self, "No Video", "Please load a video first.")
            return
        
        try:
            # Initialize video storage manager if not already done
            if not self.video_storage_manager:
                self.video_storage_manager = create_video_storage_manager()
            
            # Store the video
            self.status_bar.showMessage("Storing video...")
            stored_video = self.video_storage_manager.store_video(self.current_video_path)
            
            if stored_video:
                QMessageBox.information(self, "Success", 
                                      f"Video stored successfully!\n"
                                      f"Video ID: {stored_video.video_id}\n"
                                      f"Filename: {stored_video.filename}\n"
                                      f"Size: {stored_video.file_size / (1024*1024):.2f} MB")
                self.status_bar.showMessage(f"Video stored: {stored_video.filename}")
                # Refresh the stored videos list
                self.refresh_stored_videos()
            else:
                QMessageBox.critical(self, "Error", "Failed to store video.")
                self.status_bar.showMessage("Failed to store video")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error storing video: {e}")
            self.status_bar.showMessage("Error storing video")
    
    def manage_stored_videos(self):
        """Open dialog to manage stored videos."""
        try:
            # Initialize video storage manager if not already done
            if not self.video_storage_manager:
                self.video_storage_manager = create_video_storage_manager()
            
            # Get all stored videos
            videos = self.video_storage_manager.get_all_stored_videos()
            
            # Create dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Manage Stored Videos")
            dialog.setMinimumSize(800, 600)
            
            layout = QVBoxLayout(dialog)
            
            # Storage info
            storage_info = self.video_storage_manager.get_storage_info()
            info_label = QLabel(
                f"Total Videos: {storage_info.get('total_videos', 0)} | "
                f"Total Size: {storage_info.get('total_size_mb', 0):.2f} MB | "
                f"Storage Directory: {storage_info.get('storage_directory', 'N/A')}"
            )
            info_label.setStyleSheet("background-color: #f0f0f0; padding: 5px;")
            layout.addWidget(info_label)
            
            # Video list
            list_widget = QListWidget()
            
            for video in videos:
                item_text = (
                    f"ID: {video.video_id} | "
                    f"{video.filename} | "
                    f"{video.file_size / (1024*1024):.2f} MB | "
                    f"{video.width}x{video.height} | "
                    f"{video.duration:.1f}s"
                )
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, video.video_id)
                list_widget.addItem(item)
            
            layout.addWidget(list_widget)
            
            # Buttons
            button_layout = QHBoxLayout()
            
            load_btn = QPushButton("Load Selected")
            delete_btn = QPushButton("Delete Selected")
            cleanup_btn = QPushButton("Cleanup Orphaned")
            close_btn = QPushButton("Close")
            
            button_layout.addWidget(load_btn)
            button_layout.addWidget(delete_btn)
            button_layout.addWidget(cleanup_btn)
            button_layout.addWidget(close_btn)
            
            layout.addLayout(button_layout)
            
            # Button connections
            def load_selected():
                selected_items = list_widget.selectedItems()
                if selected_items:
                    video_id = selected_items[0].data(Qt.UserRole)
                    video_path = self.video_storage_manager.get_video_path(video_id)
                    if video_path:
                        self.load_video_from_path(video_path)
                        dialog.accept()
                    else:
                        QMessageBox.warning(dialog, "Error", "Video file not found.")
                else:
                    QMessageBox.warning(dialog, "No Selection", "Please select a video to load.")
            
            def delete_selected():
                selected_items = list_widget.selectedItems()
                if selected_items:
                    reply = QMessageBox.question(dialog, "Confirm Delete", 
                                              "Are you sure you want to delete the selected video?",
                                              QMessageBox.Yes | QMessageBox.No)
                    if reply == QMessageBox.Yes:
                        video_id = selected_items[0].data(Qt.UserRole)
                        if self.video_storage_manager.delete_stored_video(video_id):
                            list_widget.takeItem(list_widget.row(selected_items[0]))
                            # Update storage info
                            updated_info = self.video_storage_manager.get_storage_info()
                            info_label.setText(
                                f"Total Videos: {updated_info.get('total_videos', 0)} | "
                                f"Total Size: {updated_info.get('total_size_mb', 0):.2f} MB | "
                                f"Storage Directory: {updated_info.get('storage_directory', 'N/A')}"
                            )
                            QMessageBox.information(dialog, "Success", "Video deleted successfully.")
                        else:
                            QMessageBox.critical(dialog, "Error", "Failed to delete video.")
                else:
                    QMessageBox.warning(dialog, "No Selection", "Please select a video to delete.")
            
            def cleanup_orphaned():
                reply = QMessageBox.question(dialog, "Confirm Cleanup", 
                                          "This will remove files that exist in storage but not in database. Continue?",
                                          QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    deleted_count = self.video_storage_manager.cleanup_orphaned_files()
                    QMessageBox.information(dialog, "Cleanup Complete", 
                                          f"Cleaned up {deleted_count} orphaned files.")
                    # Refresh the dialog
                    dialog.accept()
                    self.manage_stored_videos()
            
            load_btn.clicked.connect(load_selected)
            delete_btn.clicked.connect(delete_selected)
            cleanup_btn.clicked.connect(cleanup_orphaned)
            close_btn.clicked.connect(dialog.accept)
            
            # Show dialog
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error managing stored videos: {e}")
    
    def refresh_stored_videos(self):
        """Refresh the stored videos list in the main panel."""
        try:
            # Check if GUI is ready and visible
            if not self.isVisible():
                return
                
            # Initialize video storage manager if not already done
            if not self.video_storage_manager:
                self.video_storage_manager = create_video_storage_manager()
            
            # Get all stored videos
            videos = self.video_storage_manager.get_all_stored_videos()
            
            # Sort videos based on current selection
            if hasattr(self, 'stored_videos_sort_combo') and self.stored_videos_sort_combo:
                sort_by = self.stored_videos_sort_combo.currentText().lower()
                videos = self._sort_videos(videos, sort_by)
            
            # Check if stored_videos_list still exists
            if hasattr(self, 'stored_videos_list') and self.stored_videos_list:
                self.stored_videos_list.clear()
                
                if videos:
                    for video in videos:
                        # Format video info for display
                        size_mb = video.file_size / (1024 * 1024)
                        duration_str = self._format_duration(video.duration)
                        item_text = f"{video.filename} ({size_mb:.1f}MB, {duration_str})"
                        
                        # Add item with video data as user role
                        item = QListWidgetItem(item_text)
                        item.setData(Qt.UserRole, video.video_id)
                        self.stored_videos_list.addItem(item)
                else:
                    self.stored_videos_list.addItem("No stored videos")
                    
                self.status_bar.showMessage(f"Found {len(videos)} stored videos")
            else:
                print("Stored videos list not available")
                
        except Exception as e:
            print(f"Error refreshing stored videos: {e}")
            self.status_bar.showMessage("Error refreshing stored videos")
    
    def load_selected_stored_video(self):
        """Load the selected stored video."""
        try:
            # Check if any item is selected
            selected_items = self.stored_videos_list.selectedItems()
            if not selected_items:
                QMessageBox.warning(self, "No Selection", "Please select a video to load.")
                return
            
            # Get video ID from selected item
            selected_item = selected_items[0]
            video_id = selected_item.data(Qt.UserRole)
            
            if video_id is None:
                QMessageBox.warning(self, "Invalid Selection", "Please select a valid video.")
                return
            
            # Initialize video storage manager if not already done
            if not self.video_storage_manager:
                self.video_storage_manager = create_video_storage_manager()
            
            # Get stored video
            stored_video = self.video_storage_manager.get_stored_video(video_id)
            if not stored_video:
                QMessageBox.critical(self, "Error", "Video not found in storage.")
                return
            
            # Update access time
            self.video_storage_manager.update_access_time(video_id)
            
            # Load video from stored path
            self.load_video_from_path(stored_video.stored_path)
            
            self.status_bar.showMessage(f"Loaded stored video: {stored_video.filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading stored video: {e}")
            self.status_bar.showMessage("Error loading stored video")
    
    def on_stored_video_selected(self):
        """Handle selection of a stored video and show details."""
        try:
            # Check if video_details_text still exists
            if not hasattr(self, 'video_details_text') or not self.video_details_text:
                return
                
            selected_items = self.stored_videos_list.selectedItems()
            if not selected_items:
                self.video_details_text.clear()
                self.video_details_text.setPlaceholderText("Select a video to view details...")
                return
            
            # Get video ID from selected item
            selected_item = selected_items[0]
            video_id = selected_item.data(Qt.UserRole)
            
            if video_id is None:
                self.video_details_text.clear()
                self.video_details_text.setPlaceholderText("Select a video to view details...")
                return
            
            # Initialize video storage manager if not already done
            if not self.video_storage_manager:
                self.video_storage_manager = create_video_storage_manager()
            
            # Get stored video
            stored_video = self.video_storage_manager.get_stored_video(video_id)
            if not stored_video:
                self.video_details_text.setText("Video details not found.")
                return
            
            # Format video details
            details_text = f"""Video ID: {stored_video.video_id}
Filename: {stored_video.filename}
Original Path: {stored_video.original_path}
Stored Path: {stored_video.stored_path}

File Size: {stored_video.file_size / (1024*1024):.2f} MB
Dimensions: {stored_video.width} x {stored_video.height}
Duration: {self._format_duration(stored_video.duration)}
FPS: {stored_video.fps:.2f}
Frame Count: {stored_video.frame_count}

Storage Date: {stored_video.storage_date}
Last Accessed: {stored_video.last_accessed}"""
            
            self.video_details_text.setText(details_text)
            
        except Exception as e:
            print(f"Error showing video details: {e}")
            self.video_details_text.setText("Error loading video details.")
    
    def _format_duration(self, duration_seconds):
        """Format duration in seconds to HH:MM:SS format."""
        if duration_seconds is None or duration_seconds <= 0:
            return "00:00:00"
            
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _sort_videos(self, videos, sort_by):
        """Sort videos based on the specified criteria."""
        if not videos:
            return videos
            
        try:
            if sort_by == "name":
                return sorted(videos, key=lambda v: v.filename.lower())
            elif sort_by == "date":
                return sorted(videos, key=lambda v: v.storage_date, reverse=True)
            elif sort_by == "size":
                return sorted(videos, key=lambda v: v.file_size, reverse=True)
            elif sort_by == "duration":
                return sorted(videos, key=lambda v: v.duration if v.duration else 0, reverse=True)
            else:
                return videos
        except Exception as e:
            print(f"Error sorting videos: {e}")
            return videos
    
    def on_stored_video_sort_changed(self):
        """Handle change in stored videos sorting."""
        try:
            self.refresh_stored_videos()
        except Exception as e:
            print(f"Error handling sort change: {e}")
    
    def load_video_from_path(self, video_path):
        """Load video from a specific path."""
        try:
            if not os.path.exists(video_path):
                QMessageBox.warning(self, "File Not Found", f"Video file not found: {video_path}")
                return
            
            # Load video using existing method
            self.current_video_path = video_path
            
            # Initialize video processor
            if self.video_processor:
                self.video_processor.release()
            
            self.video_processor = VideoProcessor(video_path)
            video_info = self.video_processor.get_video_info()
            
            if video_info:
                # Update UI
                self.setWindowTitle(f"Video Analysis - {os.path.basename(video_path)}")
                
                if hasattr(self, 'position_slider') and self.position_slider:
                    self.position_slider.setMaximum(video_info.frame_count - 1)
                
                self.update_time_display(0, video_info.frame_count)
                self.status_bar.showMessage(f"Video loaded: {video_info.width}x{video_info.height}, {video_info.fps:.2f} fps, {video_info.frame_count} frames")
                
                # Load first frame
                self.load_frame(0)
                
                # Update access time if this is a stored video
                if self.video_storage_manager:
                    # Try to find the video in storage
                    videos = self.video_storage_manager.get_all_stored_videos()
                    for video in videos:
                        if video.stored_path == video_path:
                            self.video_storage_manager.update_access_time(video.video_id)
                            break
            else:
                QMessageBox.critical(self, "Error", "Failed to load video information.")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading video: {e}")
            
    def on_session_selected(self, index):
        """Handle session selection."""
        if index == 0:  # "Select Session..."
            self.current_session_id = None
            
    def closeEvent(self, event):
        """Handle application close event."""
        try:
            # Stop video playback
            self.stop_video()
            
            # Disconnect all signals to prevent access to deleted objects
            self.disconnect_signals()
            
            # Clean up video processor
            if self.video_processor:
                self.video_processor.release()
                self.video_processor = None
            
            # Clean up thread
            if self.video_thread:
                self.video_thread.quit()
                self.video_thread.wait()
                self.video_thread = None
            
            # Clean up audio manager
            if self.audio_manager:
                self.audio_manager.stop_audio()
                self.audio_manager = None
            
            # Clean up database manager
            if self.db_manager:
                self.db_manager.close()
                self.db_manager = None
            
            # Clean up video storage manager
            if self.video_storage_manager:
                self.video_storage_manager = None
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        event.accept()
    
    def load_stored_video(self, video_id):
        """Load stored video by ID."""
        video = self.db_manager.get_stored_video(video_id)
        if video:
            self.current_video_path = video.stored_path
            self.status_bar.showMessage(f"Loading stored video: {video.filename}")
            
            # Initialize video processor
            if self.video_processor:
                self.video_processor.release()
            
            self.video_processor = VideoProcessor()
            if self.video_processor.load_video(video.stored_path):
                video_info = self.video_processor.get_video_info()
                self.position_slider.setMaximum(video_info.frame_count - 1)
                self.update_time_display(0, video_info.frame_count)
                self.status_bar.showMessage(f"Loaded stored video: {video_info.width}x{video_info.height}, {video_info.fps:.2f} fps, {video_info.frame_count} frames")
                
                # Load first frame
                frame = self.video_processor.get_frame(0)
                if frame is not None:
                    self.display_frame(frame)
                
                # Start new session
                self.start_new_session()
                
                # Update last accessed time
                self.db_manager.update_video_access_time(video_id)
            else:
                self.status_bar.showMessage(f"Failed to load stored video: {video.filename}")
        else:
            self.status_bar.showMessage(f"Video not found: {video_id}")
    
    def delete_stored_video(self, video_id):
        """Delete stored video by ID."""
        if self.db_manager.delete_stored_video(video_id):
            self.status_bar.showMessage(f"Deleted video {video_id}")
            self.refresh_video_library()
        else:
            self.status_bar.showMessage(f"Failed to delete video {video_id}")
    
    def refresh_video_library(self):
        """Refresh video library display."""
        self.video_library.refresh_videos()
    
    def edit_person_profile(self, person_id):
        """Edit person profile by ID."""
        profile = self.db_manager.get_person_profile(person_id)
        if profile:
            dialog = PersonProfileDialog(profile, self)
            if dialog.exec_() == QDialog.Accepted:
                updated_profile = dialog.get_profile()
                if self.db_manager.update_person_profile(person_id, **updated_profile):
                    self.status_bar.showMessage(f"Updated profile: {updated_profile.get('profile_name', '')}")
                    self.refresh_person_profiles()
                else:
                    self.status_bar.showMessage(f"Failed to update profile: {person_id}")
        else:
            self.status_bar.showMessage(f"Profile not found: {person_id}")
    
    def delete_person_profile(self, person_id):
        """Delete person profile by ID."""
        if self.db_manager.delete_person_profile(person_id):
            self.status_bar.showMessage(f"Deleted profile: {person_id}")
            self.refresh_person_profiles()
        else:
            self.status_bar.showMessage(f"Failed to delete profile: {person_id}")
    
    def refresh_person_profiles(self):
        """Refresh person profiles display."""
        self.person_profiles.refresh_profiles()
    
    def disconnect_signals(self):
        """Disconnect all signal connections to prevent access to deleted objects."""
        try:
            # Disconnect button signals
            if hasattr(self, 'load_video_btn') and self.load_video_btn:
                self.load_video_btn.clicked.disconnect()
            if hasattr(self, 'store_video_btn') and self.store_video_btn:
                self.store_video_btn.clicked.disconnect()
            if hasattr(self, 'manage_videos_btn') and self.manage_videos_btn:
                self.manage_videos_btn.clicked.disconnect()
            if hasattr(self, 'play_pause_btn') and self.play_pause_btn:
                self.play_pause_btn.clicked.disconnect()
            if hasattr(self, 'save_session_btn') and self.save_session_btn:
                self.save_session_btn.clicked.disconnect()
            if hasattr(self, 'load_session_btn') and self.load_session_btn:
                self.load_session_btn.clicked.disconnect()
            if hasattr(self, 'refresh_stored_btn') and self.refresh_stored_btn:
                self.refresh_stored_btn.clicked.disconnect()
            if hasattr(self, 'load_stored_btn') and self.load_stored_btn:
                self.load_stored_btn.clicked.disconnect()
            if hasattr(self, 'stored_videos_list') and self.stored_videos_list:
                self.stored_videos_list.itemSelectionChanged.disconnect()
            if hasattr(self, 'stored_videos_sort_combo') and self.stored_videos_sort_combo:
                self.stored_videos_sort_combo.currentIndexChanged.disconnect()
            if hasattr(self, 'delete_session_btn') and self.delete_session_btn:
                self.delete_session_btn.clicked.disconnect()
            
            # Disconnect slider signals
            if hasattr(self, 'position_slider') and self.position_slider:
                try:
                    self.position_slider.sliderPressed.disconnect()
                    self.position_slider.sliderReleased.disconnect()
                except:
                    pass
            
            # Disconnect threshold sliders
            if hasattr(self, 'conf_slider') and self.conf_slider:
                try:
                    self.conf_slider.valueChanged.disconnect()
                except:
                    pass
            if hasattr(self, 'iou_slider') and self.iou_slider:
                try:
                    self.iou_slider.valueChanged.disconnect()
                except:
                    pass
            
            # Disconnect session combo
            if hasattr(self, 'session_combo') and self.session_combo:
                try:
                    self.session_combo.currentIndexChanged.disconnect()
                except:
                    pass
            
            # Disconnect video library signals
            if hasattr(self, 'video_library') and self.video_library:
                try:
                    self.video_library.load_video_clicked.disconnect()
                    self.video_library.delete_video_clicked.disconnect()
                    self.video_library.refresh_clicked.disconnect()
                except:
                    pass
            
            # Disconnect person profiles signals
            if hasattr(self, 'person_profiles') and self.person_profiles:
                try:
                    self.person_profiles.edit_profile_clicked.disconnect()
                    self.person_profiles.delete_profile_clicked.disconnect()
                    self.person_profiles.refresh_clicked.disconnect()
                except:
                    pass
                    
        except Exception as e:
            print(f"Error disconnecting signals: {e}")


class PersonProfileDialog(QDialog):
    """Dialog for editing person profile information."""
    
    def __init__(self, profile, parent=None):
        super().__init__(parent)
        self.profile = profile
        self.init_ui()
        self.load_profile_data()
    
    def init_ui(self):
        """Initialize the dialog UI."""
        self.setWindowTitle("Edit Person Profile")
        self.setMinimumWidth(400)
        
        layout = QVBoxLayout()
        
        # Profile name
        name_layout = QHBoxLayout()
        name_label = QLabel("Name:")
        name_layout.addWidget(name_label)
        
        self.name_edit = QLineEdit()
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)
        
        # Description
        desc_label = QLabel("Description:")
        layout.addWidget(desc_label)
        
        self.desc_edit = QTextEdit()
        self.desc_edit.setMaximumHeight(100)
        layout.addWidget(self.desc_edit)
        
        # Statistics (read-only)
        stats_group = QGroupBox("Statistics")
        stats_layout = QFormLayout()
        
        self.first_seen_label = QLabel()
        stats_layout.addRow("First Seen:", self.first_seen_label)
        
        self.last_seen_label = QLabel()
        stats_layout.addRow("Last Seen:", self.last_seen_label)
        
        self.appearances_label = QLabel()
        stats_layout.addRow("Total Appearances:", self.appearances_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.accept)
        button_layout.addWidget(self.save_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def load_profile_data(self):
        """Load profile data into the dialog."""
        self.name_edit.setText(self.profile.profile_name)
        self.desc_edit.setText(self.profile.description)
        
        # Format statistics
        try:
            first_seen = datetime.fromisoformat(self.profile.first_seen)
            self.first_seen_label.setText(first_seen.strftime("%Y-%m-%d %H:%M:%S"))
        except:
            self.first_seen_label.setText(self.profile.first_seen)
        
        try:
            last_seen = datetime.fromisoformat(self.profile.last_seen)
            self.last_seen_label.setText(last_seen.strftime("%Y-%m-%d %H:%M:%S"))
        except:
            self.last_seen_label.setText(self.profile.last_seen)
        
        self.appearances_label.setText(str(self.profile.total_appearances))
    
    def get_profile(self):
        """Get the updated profile data from the dialog."""
        return {
            'profile_name': self.name_edit.text().strip(),
            'description': self.desc_edit.toPlainText().strip()
        }


class VideoThread(QThread):
    """Thread for video processing and playback."""
    
    frame_ready = pyqtSignal(np.ndarray)
    position_changed = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, video_processor, db_manager):
        super().__init__()
        self.video_processor = video_processor
        self.db_manager = db_manager
        self.running = False
        self.paused = False
        self.current_position = 0
        self.detection_enabled = False
        self.detector = None
        self.tracker = None
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.5
        
        # Initialize face detection and profile management
        self.face_detector = FaceDetector()
        self.profile_manager = PersonProfileManager(self.db_manager)
        
    def set_detector(self, detector):
        """Set the object detector."""
        self.detector = detector
        print(f"DEBUG: VideoThread detector set to: {detector is not None}")
        
    def set_tracker(self, tracker):
        """Set the person tracker."""
        self.tracker = tracker
        print(f"DEBUG: VideoThread tracker set to: {tracker is not None}")
        
    def set_detection_enabled(self, enabled):
        """Enable or disable detection."""
        self.detection_enabled = enabled
        
    def enable_detection(self, detector, tracker, confidence_threshold, iou_threshold):
        """Enable detection with specified detector and tracker."""
        self.detector = detector
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.detection_enabled = True
        print(f"DEBUG: Detection enabled in video thread")
        
    def disable_detection(self):
        """Disable detection."""
        self.detection_enabled = False
        print(f"DEBUG: Detection disabled in video thread")
        
    def set_confidence_threshold(self, threshold):
        """Set confidence threshold for detection."""
        self.confidence_threshold = threshold
        
    def set_iou_threshold(self, threshold):
        """Set IoU threshold for detection."""
        self.iou_threshold = threshold
        
    def stop(self):
        """Stop the video processing thread."""
        self.running = False
        
    def pause(self):
        """Pause the video processing thread."""
        self.paused = True
        
    def run(self):
        """Run the video processing thread."""
        self.running = True
        
        while self.running:
            if not self.paused:
                frame = self.video_processor.get_frame(self.current_position)
                if frame is not None:
                    # Process frame with detection if enabled
                    if self.detection_enabled and self.detector and self.tracker:
                        print(f"DEBUG: Detection enabled, processing frame {self.current_position}")
                        processed_frame = self.process_frame_with_detection(frame)
                        print(f"DEBUG: Emitting processed frame with shape: {processed_frame.shape}")
                        self.frame_ready.emit(processed_frame)
                    else:
                        print(f"DEBUG: Detection not enabled (enabled={self.detection_enabled}, detector={self.detector is not None}, tracker={self.tracker is not None})")
                        print(f"DEBUG: Emitting raw frame with shape: {frame.shape}")
                        self.frame_ready.emit(frame)
                        
                    self.position_changed.emit(self.current_position)
                    
                    # Move to next frame
                    self.current_position += 1
                    
                    # Check if we've reached the end
                    if self.current_position >= self.video_processor.get_video_info().frame_count:
                        break
                        
                else:
                    break
                    
            self.msleep(int(1000 / self.video_processor.get_video_info().fps))
            
        self.finished.emit()
        
    def process_frame_with_detection(self, frame):
        """Process frame with object detection, tracking, and face detection."""
        # Check if detector is available
        if self.detector is None:
            print("DEBUG: Detector is None in process_frame_with_detection, returning original frame")
            return frame
            
        # Perform detection
        self.detector.set_confidence_threshold(self.confidence_threshold)
        detection_result = self.detector.detect_objects(frame)
        
        if detection_result is None:
            return frame
            
        detections = detection_result.detections
        
        # Perform tracking
        if self.tracker is None:
            return frame
        tracked_objects = self.tracker.update(detections)
        
        # Draw results on frame
        result_frame = frame.copy()
        
        # Process face detection and profile creation for each detected person
        for obj_id, tracked_obj in tracked_objects.items():
            if tracked_obj.class_name == 'person':
                # Draw bounding box
                x1, y1, x2, y2 = tracked_obj.bbox
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"Person {obj_id}"
                cv2.putText(result_frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Perform face detection within person bounding box
                try:
                    print(f"DEBUG: Processing face detection for person {obj_id}")
                    # Extract person region
                    person_region = frame[y1:y2, x1:x2]
                    if person_region.size > 0:
                        print(f"DEBUG: Person region size: {person_region.size}")
                        # Detect faces in person region
                        face_detections = self.face_detector.detect_faces(person_region)
                        print(f"DEBUG: Found {len(face_detections) if face_detections else 0} faces")
                        
                        if face_detections:
                            # Process the first detected face
                            face_detection = face_detections[0]
                            print(f"DEBUG: Processing face with bbox: {face_detection.bbox}")
                            
                            # Convert face coordinates to frame coordinates
                            face_x1, face_y1, face_x2, face_y2 = face_detection.bbox
                            face_x1 += x1
                            face_y1 += y1
                            face_x2 += x1
                            face_y2 += y1
                            
                            # Draw face bounding box
                            cv2.rectangle(result_frame, (face_x1, face_y1), (face_x2, face_y2), (255, 0, 0), 2)
                            cv2.putText(result_frame, "Face", (face_x1, face_y1 - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                            
                            # Create or update person profile
                            face_region = frame[face_y1:face_y2, face_x1:face_x2]
                            if face_region.size > 0:
                                self.profile_manager.create_or_update_profile(
                                    obj_id, face_region, self.current_position, face_detection.bbox
                                )
                except Exception as e:
                    print(f"Error processing face detection for person {obj_id}: {e}")
                
        return result_frame
        
    def stop(self):
        """Stop the thread."""
        self.running = False
        self.paused = False
        
    def pause(self):
        """Pause the thread."""
        self.paused = True
        
    def resume(self):
        """Resume the thread."""
        self.paused = False
        
    def set_position(self, position):
        """Set the current position."""
        self.current_position = position
        
    def enable_detection(self, detector, tracker, confidence_threshold, iou_threshold):
        """Enable object detection."""
        self.detection_enabled = True
        self.detector = detector
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
    def disable_detection(self):
        """Disable object detection."""
        self.detection_enabled = False
        self.detector = None
        self.tracker = None


# Factory function
def create_gui_application() -> VideoAnalysisGUI:
    """Create and return the main GUI application."""
    # QApplication should be created in main.py, not here
    gui = VideoAnalysisGUI()
    return gui
