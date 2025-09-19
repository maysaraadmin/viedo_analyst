"""
Video player module refactored to use the new modular structure.
"""

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import threading
import time
from collections import deque
from typing import Optional, Dict, Any

# Import from new modules
from core.config import config
from modules.object_detection import create_detector
from modules.tracking import create_person_tracker
from modules.video_processing import VideoProcessor, VideoInfo
from modules.audio_manager import create_audio_manager

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    objects_detected = pyqtSignal(dict)
    progress_updated = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        # Initialize components using the new modules
        self.video_processor = VideoProcessor()
        self.detector = create_detector(config.default_model)
        self.tracker = create_person_tracker()
        self.audio_manager = create_audio_manager()
        
        # Legacy compatibility
        self.cap = None
        self.running = False
        self.analyze = False
        self.frame_queue = deque(maxlen=config.max_frame_queue_size)
        self.confidence_threshold = config.default_confidence_threshold
        self.person_only = False
        self.current_frame = 0
        self.total_frames = 0
        self.current_video_path = None
        self.start_time = 0.0
        
        # Set detector configuration
        self.detector.set_confidence_threshold(self.confidence_threshold)
        self.detector.set_person_only(self.person_only)
        
    def set_video_source(self, source):
        # Stop any existing audio playback
        if hasattr(self, 'audio_manager'):
            try:
                self.audio_manager.stop_audio()
            except Exception as e:
                print(f"Error stopping audio: {e}")
        
        # Use the new video processor
        if self.video_processor.capture.open(source):
            self.video_info = self.video_processor.capture.get_video_info()
            self.current_frame = 0
            self.total_frames = self.video_info.frame_count
            self.cap = self.video_processor.capture.cap  # Legacy compatibility
            self.current_video_path = source
            self.start_time = time.time()
            self.video_start_time = 0.0  # Track when video started playing
            self.audio_start_time = 0.0  # Track when audio should start
            
            # Extract audio from video
            if hasattr(self, 'audio_manager'):
                try:
                    audio_path = self.audio_manager.extract_audio_from_video(source)
                    if audio_path:
                        print(f"Audio extracted successfully: {audio_path}")
                    else:
                        print("No audio found or audio extraction failed")
                except Exception as e:
                    print(f"Error extracting audio: {e}")
        else:
            print(f"Failed to open video source: {source}")
            self.cap = None
            self.total_frames = 0
            self.current_video_path = None
        
    def toggle_analysis(self, state):
        self.analyze = state
        
    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold / 100.0
        # Update the new detector
        self.detector.set_confidence_threshold(self.confidence_threshold)
        
    def set_person_only(self, person_only):
        self.person_only = person_only
        # Update the new detector
        self.detector.set_person_only(self.person_only)
        
    def run(self):
        self.running = True
        try:
            # Get video FPS for timing calculations
            fps = self.video_info.fps if hasattr(self, 'video_info') and self.video_info else 30.0
            frame_delay = 1.0 / fps if fps > 0 else 1.0 / 30.0
            
            # Initialize audio timing
            audio_started = False
            video_start_time = None
            audio_start_offset = 0.0
            
            while self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                self.current_frame += 1
                if self.total_frames > 0:
                    progress = int((self.current_frame / self.total_frames) * 100)
                    self.progress_updated.emit(progress)
                
                # Calculate current video time
                current_video_time = (self.current_frame - 1) / fps
                
                # Start audio playback when video starts
                if not audio_started and hasattr(self, 'audio_manager') and self.current_video_path:
                    audio_info = self.audio_manager.get_audio_info(self.current_video_path)
                    if audio_info and audio_info['has_audio']:
                        # Start audio playback from beginning
                        if self.audio_manager.play_audio(self.audio_manager.current_audio_path, 0.0):
                            audio_started = True
                            video_start_time = time.time()
                            print(f"Audio started at video time: {current_video_time:.3f}s")
                
                # Synchronize audio with video position
                if audio_started and hasattr(self, 'audio_manager') and self.audio_manager.is_playing:
                    # Get current audio position
                    current_audio_pos = self.audio_manager.get_current_position()
                    
                    # Calculate expected audio position based on video time
                    expected_audio_pos = current_video_time
                    
                    # If audio is out of sync by more than 0.5 seconds, restart it
                    sync_threshold = 0.5  # Reduced threshold for better precision
                    sync_diff = abs(expected_audio_pos - current_audio_pos)
                    
                    if sync_diff > sync_threshold:
                        print(f"Audio sync issue: video={expected_audio_pos:.2f}s, audio={current_audio_pos:.2f}s, diff={sync_diff:.2f}s")
                        
                        # Stop current audio playback
                        self.audio_manager.stop_audio()
                        
                        # Restart audio from the correct position
                        if self.audio_manager.play_audio(self.audio_manager.current_audio_path, expected_audio_pos):
                            print(f"Audio resynchronized at: {expected_audio_pos:.2f}s")
                            audio_started = True
                        else:
                            print("Failed to restart audio playback")
                            audio_started = False
                    
                if self.analyze:
                    try:
                        # Use the new object detection module
                        detection_result = self.detector.detect(frame)
                        detected_objects = {}
                        
                        # Draw bounding boxes and count objects
                        for detection in detection_result.detections:
                            x1, y1, x2, y2 = int(detection.bbox.x1), int(detection.bbox.y1), int(detection.bbox.x2), int(detection.bbox.y2)
                            class_name = detection.class_name
                            confidence = detection.confidence
                            
                            # Draw bounding box
                            color = config.person_color if class_name == 'person' else config.other_object_color
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add label
                            label = f"{class_name}: {confidence:.2f}"
                            cv2.putText(frame, label, (x1, y1 - 10), 
                                       config.text_font, config.text_scale, color, config.text_thickness)
                            
                            # Count objects
                            if class_name in detected_objects:
                                detected_objects[class_name] += 1
                            else:
                                detected_objects[class_name] = 1
                        
                        self.objects_detected.emit(detected_objects)
                    except Exception as e:
                        print(f"Detection error: {e}")
                        continue
                
                self.frame_ready.emit(frame)
                time.sleep(frame_delay)  # Use calculated frame delay for better sync
        except Exception as e:
            print(f"Video thread error: {e}")
        finally:
            self.running = False
            # Stop audio when video stops
            if hasattr(self, 'audio_manager'):
                self.audio_manager.stop_audio()
            
    def stop(self):
        self.running = False
        
        # Stop audio playback with error handling
        if hasattr(self, 'audio_manager'):
            try:
                self.audio_manager.stop_audio()
            except Exception as e:
                print(f"Error stopping audio: {e}")
        
        # Use the new video processor
        try:
            self.video_processor.capture.close()
        except Exception as e:
            print(f"Error closing video processor: {e}")
        
        self.cap = None  # Legacy compatibility
        self.wait()  # Wait for thread to finish

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(config.window_title)
        self.setGeometry(*config.window_geometry)
        
        # Initialize video thread
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.objects_detected.connect(self.update_object_count)
        self.video_thread.progress_updated.connect(self.update_progress)
        
        self.init_ui()
        
    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Video display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Video label
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.video_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Video")
        self.load_btn.clicked.connect(self.load_video)
        control_layout.addWidget(self.load_btn)
        
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_video)
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)
        
        # Audio controls
        self.mute_btn = QPushButton("🔊")
        self.mute_btn.clicked.connect(self.toggle_mute)
        self.mute_btn.setEnabled(False)
        self.mute_btn.setMaximumWidth(40)
        control_layout.addWidget(self.mute_btn)
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)  # Default volume 70%
        self.volume_slider.valueChanged.connect(self.update_volume)
        self.volume_slider.setEnabled(False)
        self.volume_slider.setMaximumWidth(100)
        control_layout.addWidget(self.volume_slider)
        
        left_layout.addLayout(control_layout)
        
        # Right panel - Analysis and controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Analysis toggle
        self.analysis_cb = QCheckBox("Enable Object Detection")
        self.analysis_cb.stateChanged.connect(self.toggle_analysis)
        right_layout.addWidget(self.analysis_cb)
        
        # Object detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()
        
        self.person_only_cb = QCheckBox("Person Detection Only")
        self.person_only_cb.stateChanged.connect(self.toggle_person_only)
        detection_layout.addWidget(self.person_only_cb)
        
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 99)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.update_confidence_threshold)
        detection_layout.addWidget(QLabel("Confidence Threshold:"))
        detection_layout.addWidget(self.confidence_slider)
        
        detection_group.setLayout(detection_layout)
        right_layout.addWidget(detection_group)
        
        # Object count display
        self.object_count_text = QTextEdit()
        self.object_count_text.setMaximumHeight(200)
        self.object_count_text.setReadOnly(True)
        right_layout.addWidget(QLabel("Detected Objects:"))
        right_layout.addWidget(self.object_count_text)
        
        # Video information
        self.video_info_text = QTextEdit()
        self.video_info_text.setMaximumHeight(150)
        self.video_info_text.setReadOnly(True)
        right_layout.addWidget(QLabel("Video Information:"))
        right_layout.addWidget(self.video_info_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)
        
        # Add stretch to push everything up
        right_layout.addStretch()
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 2)
        main_layout.addWidget(right_panel, 1)
        
        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        
        open_action = QAction('Open Video', self)
        open_action.triggered.connect(self.load_video)
        file_menu.addAction(open_action)
        
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
    def load_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm);;All Files (*)"
        )
        
        if video_path:
            self.video_thread.set_video_source(video_path)
            self.play_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            
            # Check if video has audio and enable audio controls
            if hasattr(self.video_thread, 'audio_manager'):
                try:
                    audio_info = self.video_thread.audio_manager.get_audio_info(video_path)
                    has_audio = audio_info and audio_info.get('has_audio', False)
                    self.mute_btn.setEnabled(has_audio)
                    self.volume_slider.setEnabled(has_audio)
                    
                    if has_audio:
                        print(f"Video has audio: {audio_info['duration']:.2f}s, "
                              f"sample_rate: {audio_info.get('sample_rate', 'unknown')}Hz, "
                              f"channels: {audio_info.get('nchannels', 'unknown')}")
                    else:
                        print("Video has no audio track")
                except Exception as e:
                    print(f"Error checking audio info: {e}")
                    # Disable audio controls on error
                    self.mute_btn.setEnabled(False)
                    self.volume_slider.setEnabled(False)
            
            # Get video information
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                info_text = f"Resolution: {width}x{height}\n"
                info_text += f"FPS: {fps:.2f}\n"
                info_text += f"Duration: {duration:.2f} seconds\n"
                info_text += f"Total Frames: {frame_count}"
                
                self.video_info_text.setText(info_text)
            except Exception as e:
                print(f"Error getting video info: {e}")
                self.video_info_text.setText("Error loading video information")
            
    def play_video(self):
        if not self.video_thread.isRunning():
            self.video_thread.start()
            
    def pause_video(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
            
    def stop_video(self):
        if self.video_thread.isRunning():
            self.video_thread.stop()
        self.video_label.clear()
        self.video_label.setStyleSheet("background-color: black;")
        # Disable audio controls
        self.mute_btn.setEnabled(False)
        self.volume_slider.setEnabled(False)
        
    def toggle_analysis(self, state):
        self.video_thread.toggle_analysis(state == Qt.Checked)
        
    def toggle_person_only(self, state):
        self.video_thread.set_person_only(state == Qt.Checked)
        
    def update_confidence_threshold(self, value):
        self.video_thread.set_confidence_threshold(value)
        
    def toggle_mute(self):
        if hasattr(self.video_thread, 'audio_manager'):
            try:
                muted = self.video_thread.audio_manager.toggle_mute()
                # Update mute button icon
                self.mute_btn.setText("🔇" if muted else "🔊")
            except Exception as e:
                print(f"Error toggling mute: {e}")
    
    def update_volume(self, value):
        if hasattr(self.video_thread, 'audio_manager'):
            try:
                volume = value / 100.0  # Convert to 0.0-1.0 range
                self.video_thread.audio_manager.set_volume(volume)
            except Exception as e:
                print(f"Error updating volume: {e}")
        
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_frame(self, frame):
        # Convert frame to QPixmap
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    def update_object_count(self, objects):
        try:
            count_text = ""
            person_count = objects.get('person', 0)
            if person_count > 0:
                count_text += f"👥 Persons: {person_count}\n"
                
            for obj, count in objects.items():
                if obj != 'person':
                    count_text += f"📦 {obj.capitalize()}: {count}\n"
                    
            self.object_count_text.setText(count_text)
        except Exception as e:
            print(f"Error updating object count: {e}")
            
    def closeEvent(self, event):
        try:
            self.video_thread.stop()
            # Clean up audio resources
            if hasattr(self.video_thread, 'audio_manager'):
                self.video_thread.audio_manager.cleanup()
        except Exception as e:
            print(f"Error stopping video thread: {e}")
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())