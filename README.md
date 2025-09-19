# Video Analysis Application

A modular video analysis tool with object detection, tracking, and database capabilities.

## Features

- **Object Detection**: YOLO-based real-time object detection
- **Person Tracking**: Advanced person tracking across video frames
- **Database Management**: SQLite database for storing analysis sessions
- **Modern GUI**: PyQt5-based graphical user interface
- **Modular Architecture**: Clean separation of concerns for better maintainability

## Project Structure

```
videos analysis/
├── __init__.py              # Package initialization
├── main.py                  # Main application entry point
├── config.py                # Configuration and constants
├── video_processing.py      # Video capture and processing
├── object_detection.py      # YOLO object detection
├── tracking.py              # Person tracking logic
├── database.py              # Database operations
├── ui.py                    # GUI components
├── advanced_features.py     # Legacy advanced features (to be migrated)
├── video_player.py          # Legacy video player (to be migrated)
└── requirements.txt         # Python dependencies
```

## Module Descriptions

### config.py
Contains all application settings, constants, and configuration parameters:
- Application settings (name, version, window geometry)
- Video processing settings (FPS, frame delay, queue sizes)
- Object detection settings (model, confidence threshold, colors)
- Tracking settings (max distance, disappeared frames, trajectory length)
- Database settings (path, cleanup settings)
- UI settings (colors, fonts, formats)

### video_processing.py
Handles video capture, playback, and frame management:
- `VideoCapture`: Enhanced video capture with error handling
- `VideoInfo`: Data class for video metadata
- `FrameBuffer`: Thread-safe frame buffer for smooth playback
- `VideoProcessor`: Main video processing coordinator

### object_detection.py
YOLO-based object detection with configurable settings:
- `Detection`: Data class for detection results
- `DetectionResult`: Frame-level detection results
- `YOLODetector`: Main object detector class
- `DetectionFilter`: Filter for detection results
- `DetectionStats`: Statistics for detection performance

### tracking.py
Person tracking across video frames:
- `TrackedObject`: Data class for tracked objects
- `PersonTracker`: Centroid-based person tracker
- `TrackingStats`: Statistics for tracking performance

### database.py
SQLite database operations for data persistence:
- `AnalysisSession`: Data class for analysis sessions
- `FrameDetection`: Data class for frame detection data
- `DatabaseManager`: Main database operations class
- Support for sessions, frame detections, and tracked objects

### ui.py
PyQt5-based GUI components:
- `VideoDisplayWidget`: Video frame display
- `ControlPanel`: Video and analysis controls
- `InfoPanel`: Information display panels
- `SessionManagerWidget`: Analysis session management
- `VideoAnalysisGUI`: Main application window

### main.py
Application entry point:
- Dependency checking
- Component initialization
- Logging setup
- GUI application creation

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### GUI Mode
Run the application with graphical interface:
```bash
python main.py
```

### CLI Mode
Run in command-line mode for batch processing:
```bash
python main.py --cli
```

## Configuration

Edit `config.py` to customize:
- Detection confidence thresholds
- Tracking parameters
- Database paths
- UI appearance
- Video processing settings

## Migration from Legacy Files

The following legacy files are being phased out:
- `advanced_features.py`: Features will be migrated to respective modules
- `video_player.py`: Functionality moved to `video_processing.py` and `ui.py`

## Dependencies

- PyQt5: GUI framework
- OpenCV: Video processing and computer vision
- Ultralytics: YOLO object detection
- NumPy: Numerical computing
- SQLite3: Database operations

## Contributing

1. Follow the modular architecture
2. Add new features to appropriate modules
3. Update configuration in `config.py`
4. Maintain separation of concerns
5. Add proper error handling and logging

## License

This project is licensed under the MIT License.
