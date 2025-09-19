"""
Comprehensive log management system for video analysis application.
Provides a unified interface for all logging operations.
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable

from .config import config
from .logging_manager import log_manager, get_logger, setup_logging
from .log_rotation import log_rotation_manager, rotate_logs, cleanup_old_logs
from .log_analyzer import log_analyzer, log_monitor, analyze_all_logs, start_log_monitoring, stop_log_monitoring
from .log_viewer import LogViewer


class LogManager:
    """Unified log management system."""
    
    def __init__(self):
        self.viewer = LogViewer()
        self.monitoring_active = False
        self.alert_callbacks = []
        self._logger_initialized = False
        self._logger = None
        
    def _get_logger(self):
        """Get logger instance safely, initializing if needed."""
        if not self._logger_initialized:
            try:
                self._logger = get_logger('main')
                self._logger_initialized = True
            except Exception:
                # If logger fails to initialize, return None
                self._logger = None
        return self._logger
    
    @property
    def logger(self):
        """Get logger property with fallback."""
        return self._get_logger()
        
    def initialize_logging(self) -> bool:
        """Initialize the logging system."""
        try:
            # Create log directories if they don't exist
            log_dir = Path(config.logs_directory)
            log_dir.mkdir(exist_ok=True)
            
            # Setup logging
            setup_logging()
            
            # Reset logger initialization flag to force re-initialization
            self._logger_initialized = False
            
            # Log initialization
            logger = self.logger
            if logger:
                logger.info("Log management system initialized")
                logger.info(f"Log directory: {log_dir}")
            else:
                print("Log management system initialized (logger not available)")
                print(f"Log directory: {log_dir}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            return False
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of all log files."""
        return log_rotation_manager.get_log_summary()
    
    def rotate_logs_if_needed(self) -> int:
        """Rotate logs if they exceed size limits."""
        return rotate_logs()
    
    def cleanup_old_logs(self, days_to_keep: int = 30) -> int:
        """Clean up old log files."""
        return cleanup_old_logs()
    
    def analyze_logs(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze all log files and return comprehensive analysis."""
        return analyze_all_logs(hours_back)
    
    def export_log_report(self, output_path: Optional[str] = None) -> Optional[str]:
        """Export a comprehensive log report."""
        return log_rotation_manager.export_log_report(output_path)
    
    def start_real_time_monitoring(self):
        """Start real-time log monitoring."""
        if not self.monitoring_active:
            # Implement actual monitoring logic here
            self.monitoring_active = True
            
            # Use logger if available, otherwise print
            logger = self.logger
            if logger:
                logger.info("Real-time log monitoring started")
            else:
                print("Real-time log monitoring started")
            
            # TODO: Implement actual file monitoring logic
            # For now, just set the flag to indicate monitoring is active
    
    def stop_real_time_monitoring(self):
        """Stop real-time log monitoring."""
        if self.monitoring_active:
            # Implement actual monitoring stop logic here
            self.monitoring_active = False
            
            # Use logger if available, otherwise print
            logger = self.logger
            if logger:
                logger.info("Real-time log monitoring stopped")
            else:
                print("Real-time log monitoring stopped")
            
            # TODO: Implement actual file monitoring stop logic
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback function for log alerts."""
        log_monitor.add_alert_callback(callback)
        self.alert_callbacks.append(callback)
        
        # Use logger if available, otherwise print
        logger = self.logger
        if logger:
            logger.info(f"Added alert callback: {callback.__name__}")
        else:
            print(f"Added alert callback: {callback.__name__}")
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent log alerts."""
        return log_monitor.get_recent_alerts(count)
    
    def view_log_file(self, filename: str, lines: int = 50, 
                     level_filter: Optional[str] = None, 
                     search_pattern: Optional[str] = None,
                     follow: bool = False):
        """View a log file with filtering options."""
        self.viewer.view_log_file(filename, lines, level_filter, search_pattern, follow)
    
    def search_logs(self, pattern: str, filename: Optional[str] = None, 
                   context_lines: int = 2) -> List[Dict[str, Any]]:
        """Search for patterns in log files."""
        return self.viewer.search_logs(pattern, filename, context_lines)
    
    def export_logs(self, filename: str, output_path: str, 
                   format_type: str = 'json', 
                   level_filter: Optional[str] = None,
                   search_pattern: Optional[str] = None):
        """Export log data to a file."""
        self.viewer.export_logs(filename, output_path, format_type, level_filter, search_pattern)
    
    def list_log_files(self) -> List[Dict[str, Any]]:
        """List all available log files."""
        return self.viewer.list_log_files()
    
    def get_log_statistics(self, filename: Optional[str] = None, 
                          hours_back: int = 24) -> Dict[str, Any]:
        """Get statistics for log files."""
        return self.viewer.show_log_statistics(filename, hours_back)
    
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform routine log maintenance tasks."""
        # Prevent recursive calls
        if hasattr(self, '_maintenance_in_progress') and self._maintenance_in_progress:
            return {
                'timestamp': datetime.now().isoformat(),
                'rotated_files': 0,
                'cleaned_files': 0,
                'errors': ['Recursive maintenance call detected']
            }
        
        self._maintenance_in_progress = True
        
        maintenance_results = {
            'timestamp': datetime.now().isoformat(),
            'rotated_files': 0,
            'cleaned_files': 0,
            'errors': []
        }
        
        try:
            # Rotate logs if needed - call directly to avoid recursion
            from .log_rotation import log_rotation_manager
            rotated = log_rotation_manager.rotate_logs()
            maintenance_results['rotated_files'] = rotated
            
            # Clean up old logs - call directly to avoid recursion
            cleaned = log_rotation_manager.cleanup_old_logs()
            maintenance_results['cleaned_files'] = cleaned
            
            # Generate and export report
            report_path = self.export_log_report()
            if report_path:
                maintenance_results['report_path'] = report_path
            
            # Use print instead of logger to prevent recursion
            print(f"Log maintenance completed: {rotated} files rotated, {cleaned} files cleaned")
            
        except Exception as e:
            error_msg = f"Log maintenance error: {e}"
            maintenance_results['errors'].append(error_msg)
            
            # Use print instead of logger to prevent recursion
            print(error_msg)
        
        finally:
            self._maintenance_in_progress = False
        
        return maintenance_results
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information based on logs."""
        health = {
            'timestamp': datetime.now().isoformat(),
            'status': 'healthy',
            'issues': [],
            'metrics': {}
        }
        
        # Prevent recursive calls
        if hasattr(self, '_health_check_in_progress') and self._health_check_in_progress:
            health['status'] = 'error'
            health['issues'].append('Recursive health check detected')
            return health
        
        self._health_check_in_progress = True
        
        try:
            # Analyze recent logs with error handling
            try:
                analysis = self.analyze_logs(hours_back=1)
            except Exception as analysis_error:
                health['status'] = 'degraded'
                health['issues'].append(f'Log analysis failed: {analysis_error}')
                return health
            
            # Check error rates
            total_errors = analysis['summary']['total_errors']
            if total_errors > 10:
                health['status'] = 'degraded'
                health['issues'].append(f"High error rate: {total_errors} errors in the last hour")
            
            # Check for critical errors
            for filename, file_analysis in analysis['file_analyses'].items():
                if 'CRITICAL' in str(file_analysis):
                    health['status'] = 'critical'
                    health['issues'].append(f"Critical errors found in {filename}")
            
            # Get log file metrics with error handling
            try:
                summary = self.get_log_summary()
                health['metrics'] = {
                    'total_log_files': summary['total_files'],
                    'total_log_size_mb': summary['total_size_mb'],
                    'largest_log_file_mb': max([f['size_mb'] for f in summary['files']], default=0)
                }
                
                # Check log file sizes
                if health['metrics']['largest_log_file_mb'] > 100:
                    health['issues'].append(f"Large log file detected: {health['metrics']['largest_log_file_mb']:.2f}MB")
            except Exception as metrics_error:
                health['issues'].append(f'Failed to get log metrics: {metrics_error}')
            
            # Use print instead of logger to avoid recursion
            print(f"System health check completed: {health['status']}")
            
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Health check failed: {e}")
            print(f"System health check failed: {e}")
        
        finally:
            self._health_check_in_progress = False
        
        return health
    
    def create_log_backup(self, backup_path: Optional[str] = None) -> Optional[str]:
        """Create a backup of all log files."""
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = Path(config.logs_directory) / f"logs_backup_{timestamp}"
        
        backup_dir = Path(backup_path)
        
        try:
            # Create backup directory with proper error handling
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if source directory exists
            source_dir = Path(config.logs_directory)
            if not source_dir.exists():
                raise FileNotFoundError(f"Source log directory not found: {source_dir}")
            
            # Copy all log files with individual error handling
            files_copied = 0
            files_failed = 0
            
            for log_file in source_dir.glob('*.log*'):
                try:
                    import shutil
                    shutil.copy2(log_file, backup_dir)
                    files_copied += 1
                except Exception as e:
                    files_failed += 1
                    print(f"Warning: Failed to copy log file {log_file}: {e}")
                    continue
            
            # Create backup manifest with error handling
            try:
                manifest = {
                    'backup_time': datetime.now().isoformat(),
                    'backup_path': str(backup_dir),
                    'files_copied': files_copied,
                    'files_failed': files_failed,
                    'source_directory': str(config.logs_directory)
                }
                
                with open(backup_dir / 'backup_manifest.json', 'w') as f:
                    json.dump(manifest, f, indent=2)
            except Exception as e:
                print(f"Warning: Failed to create backup manifest: {e}")
            
            # Log backup result
            backup_message = f"Log backup created: {backup_dir} ({files_copied} files copied"
            if files_failed > 0:
                backup_message += f", {files_failed} files failed)"
            else:
                backup_message += ")"
            
            # Use logger if available, otherwise print
            logger = self.logger
            if logger:
                logger.info(backup_message)
            else:
                print(backup_message)
            
            return str(backup_dir)
            
        except Exception as e:
            # Use logger if available, otherwise print
            error_msg = f"Failed to create log backup: {e}"
            logger = self.logger
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            
            return None
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore log files from a backup."""
        backup_dir = Path(backup_path)
        
        # Validate backup directory
        if not backup_dir.exists():
            error_msg = f"Backup directory not found: {backup_path}"
            logger = self.logger
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return False
        
        if not backup_dir.is_dir():
            error_msg = f"Backup path is not a directory: {backup_path}"
            logger = self.logger
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return False
        
        try:
            # Check for backup manifest
            manifest_file = backup_dir / 'backup_manifest.json'
            if not manifest_file.exists():
                error_msg = "Backup manifest not found"
                logger = self.logger
                if logger:
                    logger.error(error_msg)
                else:
                    print(error_msg)
                return False
            
            # Ensure log directory exists
            log_dir = Path(config.logs_directory)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Restore log files with individual error handling
            files_restored = 0
            files_failed = 0
            
            for backup_file in backup_dir.glob('*.log*'):
                if backup_file.name != 'backup_manifest.json':
                    try:
                        import shutil
                        shutil.copy2(backup_file, log_dir)
                        files_restored += 1
                    except Exception as e:
                        files_failed += 1
                        print(f"Warning: Failed to restore log file {backup_file}: {e}")
                        continue
            
            # Log restore result
            restore_message = f"Log files restored from backup: {backup_path} ({files_restored} files restored"
            if files_failed > 0:
                restore_message += f", {files_failed} files failed)"
            else:
                restore_message += ")"
            
            # Use logger if available, otherwise print
            logger = self.logger
            if logger:
                logger.info(restore_message)
            else:
                print(restore_message)
            
            return True
            
        except Exception as e:
            # Use logger if available, otherwise print
            error_msg = f"Failed to restore from backup: {e}"
            logger = self.logger
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
            return False


# Global log manager instance
log_manager_instance = LogManager()


# Convenience functions for easy access
def initialize_logging() -> bool:
    """Initialize the logging system."""
    return log_manager_instance.initialize_logging()


def get_log_summary() -> Dict[str, Any]:
    """Get a summary of all log files."""
    return log_manager_instance.get_log_summary()


def rotate_logs_if_needed() -> int:
    """Rotate logs if they exceed size limits."""
    return log_manager_instance.rotate_logs_if_needed()


def cleanup_old_logs(days_to_keep: int = 30) -> int:
    """Clean up old log files."""
    return log_manager_instance.cleanup_old_logs(days_to_keep)


def analyze_logs(hours_back: int = 24) -> Dict[str, Any]:
    """Analyze all log files."""
    return log_manager_instance.analyze_logs(hours_back)


def start_log_monitoring():
    """Start real-time log monitoring."""
    log_manager_instance.start_real_time_monitoring()

def stop_log_monitoring():
    """Stop real-time log monitoring."""
    log_manager_instance.stop_real_time_monitoring()


def perform_log_maintenance() -> Dict[str, Any]:
    """Perform routine log maintenance tasks."""
    return log_manager_instance.perform_maintenance()


def get_system_health() -> Dict[str, Any]:
    """Get system health information based on logs."""
    return log_manager_instance.get_system_health()


def create_log_backup(backup_path: Optional[str] = None) -> Optional[str]:
    """Create a backup of all log files."""
    return log_manager_instance.create_log_backup(backup_path)


def restore_from_backup(backup_path: str) -> bool:
    """Restore log files from a backup."""
    return log_manager_instance.restore_from_backup(backup_path)


# Example usage and testing
if __name__ == '__main__':
    print("Video Analysis Log Manager")
    print("=" * 40)
    
    # Initialize logging
    if initialize_logging():
        print("✓ Logging system initialized")
        
        # Get log summary
        summary = get_log_summary()
        print(f"✓ Found {summary['total_files']} log files ({summary['total_size_mb']:.2f} MB)")
        
        # Perform maintenance
        maintenance = perform_log_maintenance()
        print(f"✓ Maintenance completed: {maintenance['rotated_files']} rotated, {maintenance['cleaned_files']} cleaned")
        
        # Get system health
        health = get_system_health()
        print(f"✓ System health: {health['status']}")
        
        if health['issues']:
            print("  Issues:")
            for issue in health['issues']:
                print(f"    - {issue}")
        
        # Create backup
        backup_path = create_log_backup()
        if backup_path:
            print(f"✓ Backup created: {backup_path}")
        
        print("\nLog management system is ready!")
    else:
        print("✗ Failed to initialize logging system")
