"""
Main entry point for the Video Analysis application.
This module coordinates all other modules and provides the main application interface.
"""

import sys
import os
from typing import Optional

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import config
from core.log_manager import initialize_logging, get_log_summary, perform_log_maintenance, get_system_health
from modules.ui import create_gui_application
from modules.database import create_database_manager
from modules.video_processing import VideoProcessor
from modules.object_detection import create_detector
from modules.tracking import create_person_tracker
from modules.audio_manager import create_audio_manager
from PyQt5.QtWidgets import QApplication


def setup_logging():
    """Setup logging configuration using the new logging system."""
    if initialize_logging():
        # Reset logs at startup
        from core.logging_manager import reset_logs, get_logger
        reset_logs()
        
        logger = get_logger('main')
        logger.info("Video Analysis Application starting...")
        
        # Log system information
        log_summary = get_log_summary()
        logger.info(f"Log system initialized with {log_summary['total_files']} log files")
        
        # Perform initial maintenance
        maintenance_results = perform_log_maintenance()
        if maintenance_results['rotated_files'] > 0 or maintenance_results['cleaned_files'] > 0:
            logger.info(f"Log maintenance: {maintenance_results['rotated_files']} files rotated, {maintenance_results['cleaned_files']} files cleaned")
        
        # Check system health
        health = get_system_health()
        logger.info(f"System health status: {health['status']}")
        if health['issues']:
            for issue in health['issues']:
                logger.warning(f"System health issue: {issue}")
        
        return logger
    else:
        print("Failed to initialize logging system")
        return None


def check_dependencies():
    """Check if all required dependencies are available."""
    from core.logging_manager import get_logger
    logger = get_logger('main')
    
    required_modules = [
        'cv2', 'numpy', 'PyQt5', 'ultralytics', 'sqlite3'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.debug(f"Module {module} is available")
        except ImportError as e:
            missing_modules.append(module)
            logger.error(f"Module {module} is missing: {e}")
    
    if missing_modules:
        logger.error(f"Missing required modules: {missing_modules}")
        return False
    
    logger.info("All required dependencies are available")
    return True


def initialize_components():
    """Initialize all application components."""
    from core.logging_manager import get_logger
    logger = get_logger('main')
    
    try:
        # Initialize database manager
        logger.info("Initializing database manager...")
        db_manager = create_database_manager(config.db_path)
        
        # Initialize video processor
        logger.info("Initializing video processor...")
        video_processor = VideoProcessor()
        
        # Initialize object detector
        logger.info("Initializing object detector...")
        detector = create_detector(config.default_model)
        
        # Initialize person tracker
        logger.info("Initializing person tracker...")
        tracker = create_person_tracker()
        
        # Initialize audio manager
        logger.info("Initializing audio manager...")
        audio_manager = create_audio_manager()
        
        logger.info("All components initialized successfully")
        return {
            'db_manager': db_manager,
            'video_processor': video_processor,
            'detector': detector,
            'tracker': tracker,
            'audio_manager': audio_manager
        }
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return None


def create_application_directories():
    """Create necessary application directories."""
    from core.logging_manager import get_logger
    logger = get_logger('main')
    
    directories = [
        os.path.dirname(config.db_path),
        os.path.dirname(config.log_file_path),
        config.output_directory,
        config.temp_directory
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.debug(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")


def main():
    """Main application entry point."""
    # Setup logging
    logger = setup_logging()
    if not logger:
        print("Failed to setup logging. Exiting.")
        return
    
    logger.info("=" * 50)
    logger.info(f"Starting {config.APP_NAME} v{config.APP_VERSION}")
    logger.info("=" * 50)
    
    # Create application directories
    create_application_directories()
    
    # Check dependencies
    if not check_dependencies():
        logger.error("Missing dependencies. Please install required packages.")
        sys.exit(1)
    
    # Initialize components
    components = initialize_components()
    if not components:
        logger.error("Failed to initialize components")
        sys.exit(1)
    
    try:
        # Create QApplication instance first
        qt_app = QApplication(sys.argv)
        
        # Create and show GUI
        logger.info("Creating GUI application...")
        app = create_gui_application()
        
        # Inject components into the GUI
        app.video_processor = components['video_processor']
        app.detector = components['detector']
        app.tracker = components['tracker']
        app.db_manager = components['db_manager']
        
        # Show the application first
        app.show()
        
        # Then load initial sessions after GUI is ready
        app.refresh_sessions()
        
        logger.info("Application started successfully")
        
        # Start the event loop
        sys.exit(qt_app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    
    finally:
        # Cleanup
        logger.info("Cleaning up resources...")
        if components:
            try:
                components['video_processor'].release()
                logger.debug("Video processor released")
            except Exception as e:
                logger.error(f"Error releasing video processor: {e}")
            
            try:
                components['audio_manager'].cleanup()
                logger.debug("Audio manager cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up audio manager: {e}")
        
        logger.info("Application shutdown complete")


def run_cli_mode():
    """Run application in CLI mode for batch processing."""
    logger = setup_logging()
    if not logger:
        print("Failed to setup logging. Exiting.")
        return
    
    logger.info("Running in CLI mode")
    
    # Initialize components
    components = initialize_components()
    if not components:
        logger.error("Failed to initialize components")
        return False
    
    # TODO: Implement CLI processing logic
    logger.info("CLI mode not yet implemented")
    return True


if __name__ == "__main__":
    # Check if running in CLI mode
    cli_mode = '--cli' in sys.argv or '-c' in sys.argv
    
    if cli_mode:
        success = run_cli_mode()
        sys.exit(0 if success else 1)
    else:
        main()
