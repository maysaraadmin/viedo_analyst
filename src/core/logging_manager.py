"""
Advanced logging manager for video analysis application.
Provides structured logging with multiple log files, rotation, and different log levels.
"""

import logging
import logging.handlers
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import traceback

from .config import config


class VideoAnalysisLogManager:
    """Advanced logging manager with multiple log files and structured logging."""
    
    def __init__(self):
        self.loggers: Dict[str, logging.Logger] = {}
        self.log_dir = Path(config.logs_directory)
        self.log_dir.mkdir(exist_ok=True)
        
        # Define log files and their purposes
        self.log_files = {
            'main': 'video_analysis.log',
            'video_processing': 'video_processing.log',
            'face_detection': 'face_detection.log',
            'database': 'database.log',
            'ui_events': 'ui_events.log',
            'performance': 'performance.log',
            'errors': 'errors.log'
        }
        
        # Define log levels for different components
        self.log_levels = {
            'main': logging.INFO,
            'video_processing': logging.INFO,
            'face_detection': logging.DEBUG,
            'database': logging.INFO,
            'ui_events': logging.DEBUG,
            'performance': logging.INFO,
            'errors': logging.WARNING
        }
        
        self.setup_all_loggers()
    
    def setup_all_loggers(self):
        """Setup all loggers with appropriate handlers and formatters."""
        for logger_name, log_file in self.log_files.items():
            self.setup_logger(logger_name, log_file)
    
    def setup_logger(self, logger_name: str, log_file: str):
        """Setup a specific logger with file and console handlers."""
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.log_levels.get(logger_name, logging.INFO))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler with rotation
        file_path = self.log_dir / log_file
        try:
            # Ensure log directory exists
            self.log_dir.mkdir(exist_ok=True)
            
            # Create file handler with proper error handling
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            
            # File formatter
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Failed to setup file handler for {logger_name}: {e}")
            # Fallback to console handler only
            pass
        
        # Console handler (only for main logger)
        if logger_name == 'main':
            try:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(logging.INFO)
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
            except Exception as e:
                print(f"Warning: Failed to setup console handler for {logger_name}: {e}")
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self.loggers[logger_name] = logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger by name."""
        if name not in self.loggers:
            # Create a new logger if it doesn't exist
            self.setup_logger(name, f"{name}.log")
        return self.loggers.get(name, logging.getLogger(name))
    
    def log_structured(self, logger_name: str, level: str, message: str, **kwargs):
        """Log structured data with additional context."""
        logger = self.get_logger(logger_name)
        
        # Create structured log entry
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        # Convert to JSON string for structured logging
        structured_message = json.dumps(log_data, default=str)
        
        # Log with appropriate level
        if level.upper() == 'DEBUG':
            logger.debug(structured_message)
        elif level.upper() == 'INFO':
            logger.info(structured_message)
        elif level.upper() == 'WARNING':
            logger.warning(structured_message)
        elif level.upper() == 'ERROR':
            logger.error(structured_message)
        elif level.upper() == 'CRITICAL':
            logger.critical(structured_message)
    
    def log_error_with_traceback(self, logger_name: str, error: Exception, context: Dict[str, Any] = None):
        """Log error with full traceback and context."""
        logger = self.get_logger(logger_name)
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        logger.error(f"Error occurred: {json.dumps(error_data, default=str)}")
        
        # Also log to errors.log
        error_logger = self.get_logger('errors')
        error_logger.error(f"Error occurred: {json.dumps(error_data, default=str)}")
    
    def log_performance_metric(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        self.log_structured(
            'performance',
            'INFO',
            f'Performance metric: {operation}',
            operation=operation,
            duration_ms=duration * 1000,
            **kwargs
        )
    
    def log_ui_event(self, event_type: str, component: str, **kwargs):
        """Log UI events."""
        self.log_structured(
            'ui_events',
            'DEBUG',
            f'UI event: {event_type}',
            event_type=event_type,
            component=component,
            **kwargs
        )
    
    def log_database_operation(self, operation: str, table: str, **kwargs):
        """Log database operations."""
        self.log_structured(
            'database',
            'INFO',
            f'Database operation: {operation}',
            operation=operation,
            table=table,
            **kwargs
        )
    
    def log_video_processing_event(self, event_type: str, video_path: str, **kwargs):
        """Log video processing events."""
        self.log_structured(
            'video_processing',
            'INFO',
            f'Video processing: {event_type}',
            event_type=event_type,
            video_path=video_path,
            **kwargs
        )
    
    def log_face_detection_event(self, event_type: str, frame_number: int, **kwargs):
        """Log face detection events."""
        self.log_structured(
            'face_detection',
            'DEBUG',
            f'Face detection: {event_type}',
            event_type=event_type,
            frame_number=frame_number,
            **kwargs
        )
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old log files."""
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        for log_file in self.log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_date:
                try:
                    log_file.unlink()
                    self.get_logger('main').info(f"Cleaned up old log file: {log_file}")
                except Exception as e:
                    self.get_logger('main').error(f"Failed to clean up log file {log_file}: {e}")
    
    def reset_logs(self):
        """Reset all log files by deleting them and recreating empty ones."""
        try:
            # Close all existing handlers to release file locks
            # Close handlers for our managed loggers
            for logger_name, logger in self.loggers.items():
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            
            # Close root logger handlers
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
            
            # Close handlers for any other existing loggers
            for logger_name in logging.Logger.manager.loggerDict:
                if logger_name not in self.loggers:
                    other_logger = logging.getLogger(logger_name)
                    for handler in other_logger.handlers[:]:
                        handler.close()
                        other_logger.removeHandler(handler)
            
            # Delete all existing log files with retry logic
            for log_file in self.log_dir.glob('*.log*'):
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        log_file.unlink()
                        break
                    except Exception as e:
                        if attempt == max_retries - 1:
                            print(f"Warning: Failed to delete log file {log_file}: {e}")
                        else:
                            import time
                            time.sleep(0.1)  # Wait before retry
            
            # Recreate all loggers with fresh files
            self.setup_all_loggers()
            
            print("All log files have been reset successfully")
            return True
        except Exception as e:
            print(f"Error resetting logs: {e}")
            return False
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about log files."""
        stats = {}
        
        for logger_name, log_file in self.log_files.items():
            file_path = self.log_dir / log_file
            if file_path.exists():
                stat = file_path.stat()
                stats[logger_name] = {
                    'file_path': str(file_path),
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'exists': True
                }
            else:
                stats[logger_name] = {
                    'file_path': str(file_path),
                    'exists': False
                }
        
        return stats


# Global logging manager instance
log_manager = VideoAnalysisLogManager()


def get_logger(name: str = 'main') -> logging.Logger:
    """Get a logger instance."""
    return log_manager.get_logger(name)


def setup_logging():
    """Setup logging for the application."""
    logger = get_logger('main')
    logger.info("Video Analysis Application starting...")
    logger.info("=" * 50)
    return logger


# Convenience functions for different types of logging
def log_error(error: Exception, context: Dict[str, Any] = None, logger_name: str = 'main'):
    """Log an error with traceback."""
    log_manager.log_error_with_traceback(logger_name, error, context)


def log_performance(operation: str, duration: float, **kwargs):
    """Log a performance metric."""
    log_manager.log_performance_metric(operation, duration, **kwargs)


def log_ui_event(event_type: str, component: str, **kwargs):
    """Log a UI event."""
    log_manager.log_ui_event(event_type, component, **kwargs)


def log_database_operation(operation: str, table: str, **kwargs):
    """Log a database operation."""
    log_manager.log_database_operation(operation, table, **kwargs)


def log_video_processing_event(event_type: str, video_path: str, **kwargs):
    """Log a video processing event."""
    log_manager.log_video_processing_event(event_type, video_path, **kwargs)


def log_face_detection_event(event_type: str, frame_number: int, **kwargs):
    """Log a face detection event."""
    log_manager.log_face_detection_event(event_type, frame_number, **kwargs)


def reset_logs():
    """Reset all log files by deleting them and recreating empty ones."""
    return log_manager.reset_logs()
