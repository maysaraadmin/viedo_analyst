"""
UI module for GUI components and user interface management.
"""

import sys
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Callable
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSlider, 
                             QFileDialog, QProgressBar, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QCheckBox, QSpinBox,
                             QGroupBox, QTabWidget, QMessageBox, QSplitter)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon

from config import config
from video_processing import VideoProcessor, VideoInfo
from object_detection import YOLODetector, DetectionResult
from tracking import PersonTracker, TrackedObject
from database import DatabaseManager, AnalysisSession, StoredVideo


class VideoDisplayWidget(QWidget):
    """Widget for displaying video frames."""
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black;")
        self.current_frame = None
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setText("No video loaded")
        self.video_label.setStyleSheet("color: white; font-size: 16px;")
        
        layout.addWidget(self.video_label)
        self.setLayout(layout)
    
    def update_frame(self, frame: np.ndarray):
        """Update video frame display."""
        if frame is None:
            return
        
        self.current_frame = frame
        
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to QImage
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)
        
        # Scale pixmap to fit widget while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Update label
        self.video_label.setPixmap(scaled_pixmap)
    
    def resizeEvent(self, event):
        """Handle widget resize event."""
        super().resizeEvent(event)
        if self.current_frame is not None:
            self.update_frame(self.current_frame)
    
    def clear_display(self):
        """Clear video display."""
        self.video_label.clear()
        self.video_label.setText("No video loaded")
        self.video_label.setStyleSheet("color: white; font-size: 16px;")
        self.current_frame = None


class ControlPanel(QWidget):
    """Control panel for video playback and analysis controls."""
    
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    open_video_clicked = pyqtSignal()
    analyze_toggled = pyqtSignal(bool)
    confidence_changed = pyqtSignal(int)
    person_only_toggled = pyqtSignal(bool)
    tracking_toggled = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Video controls group
        video_group = QGroupBox("Video Controls")
        video_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play_clicked.emit)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_clicked.emit)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_clicked.emit)
        
        self.open_button = QPushButton("Open Video")
        self.open_button.clicked.connect(self.open_video_clicked.emit)
        
        video_layout.addWidget(self.open_button)
        video_layout.addWidget(self.play_button)
        video_layout.addWidget(self.pause_button)
        video_layout.addWidget(self.stop_button)
        video_group.setLayout(video_layout)
        
        # Analysis controls group
        analysis_group = QGroupBox("Analysis Controls")
        analysis_layout = QVBoxLayout()
        
        # Analysis toggle
        self.analyze_checkbox = QCheckBox("Enable Object Detection")
        self.analyze_checkbox.toggled.connect(self.analyze_toggled.emit)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(0)
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.confidence_changed.emit)
        self.confidence_label = QLabel("50%")
        self.confidence_slider.valueChanged.connect(
            lambda value: self.confidence_label.setText(f"{value}%")
        )
        confidence_layout.addWidget(self.confidence_slider)
        confidence_layout.addWidget(self.confidence_label)
        
        # Person only filter
        self.person_only_checkbox = QCheckBox("Detect Persons Only")
        self.person_only_checkbox.toggled.connect(self.person_only_toggled.emit)
        
        # Tracking toggle
        self.tracking_checkbox = QCheckBox("Enable Person Tracking")
        self.tracking_checkbox.toggled.connect(self.tracking_toggled.emit)
        
        analysis_layout.addWidget(self.analyze_checkbox)
        analysis_layout.addLayout(confidence_layout)
        analysis_layout.addWidget(self.person_only_checkbox)
        analysis_layout.addWidget(self.tracking_checkbox)
        analysis_group.setLayout(analysis_layout)
        
        # Progress bar
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        
        # Add all to main layout
        layout.addWidget(video_group)
        layout.addWidget(analysis_group)
        layout.addLayout(progress_layout)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def set_controls_enabled(self, enabled: bool):
        """Enable/disable control buttons."""
        self.play_button.setEnabled(enabled)
        self.pause_button.setEnabled(enabled)
        self.stop_button.setEnabled(enabled)
        self.analyze_checkbox.setEnabled(enabled)
    
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)


class InfoPanel(QWidget):
    """Information panel for displaying video info and detection results."""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Video info group
        video_info_group = QGroupBox("Video Information")
        video_info_layout = QVBoxLayout()
        
        self.video_info_text = QTextEdit()
        self.video_info_text.setReadOnly(True)
        self.video_info_text.setMaximumHeight(100)
        video_info_layout.addWidget(self.video_info_text)
        video_info_group.setLayout(video_info_layout)
        
        # Detection results group
        detection_group = QGroupBox("Detection Results")
        detection_layout = QVBoxLayout()
        
        self.detection_table = QTableWidget()
        self.detection_table.setColumnCount(2)
        self.detection_table.setHorizontalHeaderLabels(["Object", "Count"])
        self.detection_table.horizontalHeader().setStretchLastSection(True)
        detection_layout.addWidget(self.detection_table)
        detection_group.setLayout(detection_layout)
        
        # Tracking info group
        tracking_group = QGroupBox("Tracking Information")
        tracking_layout = QVBoxLayout()
        
        self.tracking_info_text = QTextEdit()
        self.tracking_info_text.setReadOnly(True)
        self.tracking_info_text.setMaximumHeight(100)
        tracking_layout.addWidget(self.tracking_info_text)
        tracking_group.setLayout(tracking_layout)
        
        # Add all to main layout
        layout.addWidget(video_info_group)
        layout.addWidget(detection_group)
        layout.addWidget(tracking_group)
        layout.addStretch()
        
        self.setLayout(layout)
    
    def update_video_info(self, video_info: VideoInfo):
        """Update video information display."""
        if video_info:
            info_text = f"Resolution: {video_info.width}x{video_info.height}\n"
            info_text += f"FPS: {video_info.fps:.2f}\n"
            info_text += f"Duration: {video_info.duration:.2f} seconds\n"
            info_text += f"Total Frames: {video_info.frame_count}"
            self.video_info_text.setText(info_text)
        else:
            self.video_info_text.setText("No video loaded")
    
    def update_detection_results(self, frame_count: Dict[str, int]):
        """Update detection results table."""
        self.detection_table.setRowCount(len(frame_count))
        
        for i, (class_name, count) in enumerate(frame_count.items()):
            self.detection_table.setItem(i, 0, QTableWidgetItem(class_name))
            self.detection_table.setItem(i, 1, QTableWidgetItem(str(count)))
    
    def update_tracking_info(self, tracked_objects: Dict[int, TrackedObject]):
        """Update tracking information display."""
        if tracked_objects:
            info_text = f"Tracked Objects: {len(tracked_objects)}\n"
            info_text += f"Persons: {len([obj for obj in tracked_objects.values() if obj.class_name == 'person'])}\n"
            
            if tracked_objects:
                avg_lifetime = np.mean([
                    (obj.last_seen - obj.first_seen).total_seconds() 
                    for obj in tracked_objects.values()
                ])
                info_text += f"Avg Lifetime: {avg_lifetime:.1f}s"
            
            self.tracking_info_text.setText(info_text)
        else:
            self.tracking_info_text.setText("No active tracking")


class SessionManagerWidget(QWidget):
    """Widget for managing analysis sessions."""
    
    load_session_clicked = pyqtSignal(int)
    delete_session_clicked = pyqtSignal(int)
    export_session_clicked = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        # Sessions table
        self.sessions_table = QTableWidget()
        self.sessions_table.setColumnCount(5)
        self.sessions_table.setHorizontalHeaderLabels([
            "Session ID", "Video Path", "Analysis Date", 
            "Total Frames", "Persons Detected"
        ])
        self.sessions_table.horizontalHeader().setStretchLastSection(True)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_sessions)
        
        self.load_button = QPushButton("Load Session")
        self.load_button.clicked.connect(self.load_selected_session)
        
        self.delete_button = QPushButton("Delete Session")
        self.delete_button.clicked.connect(self.delete_selected_session)
        
        self.export_button = QPushButton("Export Session")
        self.export_button.clicked.connect(self.export_selected_session)
        
        button_layout.addWidget(self.refresh_button)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addWidget(self.export_button)
        
        layout.addWidget(self.sessions_table)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def update_sessions(self, sessions: List[AnalysisSession]):
        """Update sessions table."""
        self.sessions_table.setRowCount(len(sessions))
        
        for i, session in enumerate(sessions):
            self.sessions_table.setItem(i, 0, QTableWidgetItem(str(session.session_id)))
            self.sessions_table.setItem(i, 1, QTableWidgetItem(session.video_path))
            self.sessions_table.setItem(i, 2, QTableWidgetItem(session.analysis_date))
            self.sessions_table.setItem(i, 3, QTableWidgetItem(str(session.total_frames)))
            self.sessions_table.setItem(i, 4, QTableWidgetItem(str(session.persons_detected)))
    
    def get_selected_session_id(self) -> Optional[int]:
        """Get selected session ID."""
        current_row = self.sessions_table.currentRow()
        if current_row >= 0:
            session_id_item = self.sessions_table.item(current_row, 0)
            if session_id_item:
                return int(session_id_item.text())
        return None
    
    def refresh_sessions(self):
        """Refresh sessions display."""
        # This will be connected to the main application
        pass
    
    def load_selected_session(self):
        """Load selected session."""
        session_id = self.get_selected_session_id()
        if session_id:
            self.load_session_clicked.emit(session_id)
    
    def delete_selected_session(self):
        """Delete selected session."""
        session_id = self.get_selected_session_id()
        if session_id:
            reply = QMessageBox.question(
                self, "Confirm Delete",
                f"Are you sure you want to delete session {session_id}?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.delete_session_clicked.emit(session_id)
    
    def export_selected_session(self):
        """Export selected session."""
        session_id = self.get_selected_session_id()
        if session_id:
            self.export_session_clicked.emit(session_id)


class VideoAnalysisGUI(QMainWindow):
    """Main GUI application window."""
    
    def __init__(self):
        super().__init__()
        self.video_processor = VideoProcessor()
        self.detector = YOLODetector()
        self.tracker = PersonTracker()
        self.db_manager = DatabaseManager()
        self.detection_enabled = False  # Initialize detection enabled flag
        
        self.init_ui()
        self.setup_connections()
    
    def init_ui(self):
        self.setWindowTitle(config.WINDOW_TITLE)
        self.setGeometry(*config.WINDOW_GEOMETRY)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel (video display)
        left_panel = QVBoxLayout()
        self.video_display = VideoDisplayWidget()
        left_panel.addWidget(self.video_display)
        
        # Right panel (controls and info)
        right_panel = QVBoxLayout()
        
        # Create tab widget for right panel
        tab_widget = QTabWidget()
        
        # Controls tab
        controls_tab = QWidget()
        controls_layout = QVBoxLayout()
        self.control_panel = ControlPanel()
        controls_layout.addWidget(self.control_panel)
        controls_tab.setLayout(controls_layout)
        
        # Info tab
        info_tab = QWidget()
        info_layout = QVBoxLayout()
        self.info_panel = InfoPanel()
        info_layout.addWidget(self.info_panel)
        info_tab.setLayout(info_layout)
        
        # Sessions tab
        sessions_tab = QWidget()
        sessions_layout = QVBoxLayout()
        self.session_manager = SessionManagerWidget()
        sessions_layout.addWidget(self.session_manager)
        sessions_tab.setLayout(sessions_layout)
        
        # Video Library tab
        video_library_tab = QWidget()
        video_library_layout = QVBoxLayout()
        self.video_library = VideoLibraryWidget(self.db_manager)
        video_library_layout.addWidget(self.video_library)
        video_library_tab.setLayout(video_library_layout)
        
        # Add tabs
        tab_widget.addTab(controls_tab, "Controls")
        tab_widget.addTab(info_tab, "Information")
        tab_widget.addTab(sessions_tab, "Sessions")
        tab_widget.addTab(video_library_tab, "Video Library")
        
        right_panel.addWidget(tab_widget)
        
        # Add panels to main layout
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def setup_connections(self):
        """Setup signal connections."""
        # Control panel connections
        self.control_panel.open_video_clicked.connect(self.open_video_file)
        self.control_panel.play_clicked.connect(self.play_video)
        self.control_panel.pause_clicked.connect(self.pause_video)
        self.control_panel.stop_clicked.connect(self.stop_video)
        self.control_panel.analyze_toggled.connect(self.toggle_analysis)
        self.control_panel.confidence_changed.connect(self.set_confidence_threshold)
        self.control_panel.person_only_toggled.connect(self.set_person_only)
        self.control_panel.tracking_toggled.connect(self.toggle_tracking)
        
        # Session manager connections
        self.session_manager.load_session_clicked.connect(self.load_session)
        self.session_manager.delete_session_clicked.connect(self.delete_session)
        self.session_manager.export_session_clicked.connect(self.export_session)
        
        # Video library connections
        self.video_library.load_video_clicked.connect(self.load_stored_video)
        self.video_library.delete_video_clicked.connect(self.delete_stored_video)
        self.video_library.refresh_clicked.connect(self.refresh_video_library)
        
        # Setup timer for video updates
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.update_video_display)
        self.video_timer.start(30)  # ~30 FPS
    
    def open_video_file(self):
        """Open video file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, file_path: str):
        """Load video file."""
        if self.video_processor.load_video(file_path):
            video_info = self.video_processor.get_video_info()
            self.info_panel.update_video_info(video_info)
            self.control_panel.set_controls_enabled(True)
            self.status_bar.showMessage(f"Loaded: {file_path}")
        else:
            self.status_bar.showMessage("Failed to load video")
    
    def play_video(self):
        """Start video playback."""
        if self.video_processor.is_loaded():
            self.video_processor.start_processing()
            self.status_bar.showMessage("Playing video")
    
    def pause_video(self):
        """Pause video playback."""
        self.video_processor.stop_processing()
        self.status_bar.showMessage("Video paused")
    
    def stop_video(self):
        """Stop video playback."""
        self.video_processor.stop_processing()
        self.video_display.clear_display()
        self.status_bar.showMessage("Video stopped")
    
    def toggle_analysis(self, enabled: bool):
        """Toggle object detection."""
        # Set detection enabled flag
        self.detection_enabled = enabled
        
        if enabled:
            # Ensure detector and tracker are properly initialized
            if not hasattr(self, 'detector') or self.detector is None:
                self.detector = YOLODetector()
            if not hasattr(self, 'tracker') or self.tracker is None:
                self.tracker = PersonTracker()
            
            print(f"DEBUG: Detection enabled - detector: {self.detector is not None}, tracker: {self.tracker is not None}")
        else:
            # Clear tracking data when disabling detection
            if hasattr(self, 'tracker') and self.tracker is not None:
                self.tracker.clear_all()
        
        self.status_bar.showMessage(f"Object detection {'enabled' if enabled else 'disabled'}")
    
    def set_confidence_threshold(self, value: int):
        """Set confidence threshold."""
        threshold = value / 100.0
        self.detector.set_confidence_threshold(threshold)
        self.status_bar.showMessage(f"Confidence threshold: {threshold:.2f}")
    
    def set_person_only(self, enabled: bool):
        """Set person-only detection."""
        self.detector.set_person_only(enabled)
        self.status_bar.showMessage(f"Person-only detection {'enabled' if enabled else 'disabled'}")
    
    def toggle_tracking(self, enabled: bool):
        """Toggle person tracking."""
        if not enabled:
            self.tracker.clear_all()
        self.status_bar.showMessage(f"Person tracking {'enabled' if enabled else 'disabled'}")
    
    def update_video_display(self):
        """Update video display with new frame."""
        if not self.video_processor.is_processing_active():
            return
        
        frame = self.video_processor.get_next_frame()
        if frame is not None:
            # Perform object detection if enabled
            if hasattr(self, 'detection_enabled') and self.detection_enabled:
                print(f"DEBUG: Processing frame with detection enabled")
                detection_result = self.detector.detect_objects(frame)
                if detection_result:
                    # Update detection results display
                    self.info_panel.update_detection_results(detection_result.frame_count)
                    
                    # Draw detections on frame
                    frame = self.detector.draw_detections(frame, detection_result.detections)
                    
                    # Perform tracking if enabled
                    if self.control_panel.tracking_checkbox.isChecked():
                        # Filter person detections for tracking
                        person_detections = [d for d in detection_result.detections if d.class_name == 'person']
                        tracked_objects = self.tracker.update(person_detections)
                        
                        # Update tracking info
                        self.info_panel.update_tracking_info(tracked_objects)
                        
                        # Draw tracking info
                        frame = self.tracker.draw_tracking(frame)
            else:
                print(f"DEBUG: Detection not enabled (detection_enabled={getattr(self, 'detection_enabled', False)})")
        
            # Update display
            self.video_display.update_frame(frame)
            
            # Update progress
            progress = int(self.video_processor.get_progress())
            self.control_panel.update_progress(progress)
    
    def load_session(self, session_id: int):
        """Load analysis session."""
        session = self.db_manager.get_session(session_id)
        if session:
            self.status_bar.showMessage(f"Loaded session {session_id}")
            # TODO: Implement session loading logic
        else:
            self.status_bar.showMessage(f"Failed to load session {session_id}")
    
    def delete_session(self, session_id: int):
        """Delete analysis session."""
        if self.db_manager.delete_session(session_id):
            self.status_bar.showMessage(f"Deleted session {session_id}")
            self.refresh_sessions()
        else:
            self.status_bar.showMessage(f"Failed to delete session {session_id}")
    
    def export_session(self, session_id: int):
        """Export analysis session."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Session", "", "JSON Files (*.json)"
        )
        
        if file_path:
            if self.db_manager.export_session_data(session_id, file_path):
                self.status_bar.showMessage(f"Exported session {session_id} to {file_path}")
            else:
                self.status_bar.showMessage(f"Failed to export session {session_id}")
    
    def refresh_sessions(self):
        """Refresh sessions display."""
        sessions = self.db_manager.get_all_sessions()
        self.session_manager.update_sessions(sessions)
    
    def load_stored_video(self, video_id: int):
        """Load stored video by ID."""
        video = self.db_manager.get_stored_video(video_id)
        if video:
            if self.video_processor.load_video(video.stored_path):
                video_info = self.video_processor.get_video_info()
                self.info_panel.update_video_info(video_info)
                self.control_panel.set_controls_enabled(True)
                self.status_bar.showMessage(f"Loaded stored video: {video.filename}")
                
                # Update last accessed time
                self.db_manager.update_video_access_time(video_id)
            else:
                self.status_bar.showMessage(f"Failed to load stored video: {video.filename}")
        else:
            self.status_bar.showMessage(f"Video not found: {video_id}")
    
    def delete_stored_video(self, video_id: int):
        """Delete stored video by ID."""
        if self.db_manager.delete_stored_video(video_id):
            self.status_bar.showMessage(f"Deleted video {video_id}")
            self.refresh_video_library()
        else:
            self.status_bar.showMessage(f"Failed to delete video {video_id}")
    
    def refresh_video_library(self):
        """Refresh video library display."""
        self.video_library.refresh_videos()
    
    def closeEvent(self, event):
        """Handle application close event."""
        self.video_processor.release()
        event.accept()


class VideoLibraryWidget(QWidget):
    """Widget for displaying stored videos with their information."""
    
    load_video_clicked = pyqtSignal(int)  # video_id
    delete_video_clicked = pyqtSignal(int)  # video_id
    refresh_clicked = pyqtSignal()
    
    def __init__(self, db_manager: DatabaseManager):
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
    
    def update_videos_table(self, videos: List[StoredVideo]):
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
    
    def format_duration(self, duration_seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format."""
        if duration_seconds <= 0:
            return "00:00:00"
        
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def format_file_size(self, size_bytes: int) -> str:
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
    
    def on_load_video_clicked(self, video_id: int):
        """Handle load video button click."""
        self.load_video_clicked.emit(video_id)
    
    def on_delete_video_clicked(self, video_id: int):
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


# Factory function
def create_gui_application() -> VideoAnalysisGUI:
    """Create and return the main GUI application."""
    app = QApplication(sys.argv)
    gui = VideoAnalysisGUI()
    return gui
