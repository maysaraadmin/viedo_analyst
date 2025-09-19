"""
Log rotation and management utilities for video analysis application.
Provides automated log rotation, cleanup, and management features.
"""

import os
import gzip
import shutil
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from .config import config
from .logging_manager import log_manager


class LogRotationManager:
    """Manages log rotation, compression, and cleanup."""
    
    def __init__(self):
        self.log_dir = Path(config.logs_directory)
        self.max_log_size_mb = 10  # Maximum size per log file in MB
        self.max_log_files = 5     # Maximum number of rotated log files to keep
        self.retention_days = 30   # Number of days to keep logs
        self.compression_enabled = True
        
    def rotate_logs(self):
        """Rotate all log files that exceed size limits."""
        # Avoid using logger during rotation to prevent recursion
        print("Starting log rotation process")
        
        rotated_count = 0
        
        for log_file in self.log_dir.glob('*.log'):
            if self.should_rotate_log(log_file):
                self.rotate_single_log(log_file)
                rotated_count += 1
                print(f"Rotated log file: {log_file.name}")
        
        print(f"Log rotation completed. Rotated {rotated_count} files.")
        return rotated_count
    
    def should_rotate_log(self, log_file: Path) -> bool:
        """Check if a log file should be rotated based on size."""
        if not log_file.exists():
            return False
        
        size_mb = log_file.stat().st_size / (1024 * 1024)
        return size_mb >= self.max_log_size_mb
    
    def rotate_single_log(self, log_file: Path):
        """Rotate a single log file."""
        base_name = log_file.stem
        extension = log_file.suffix
        
        # Find existing rotated files
        rotated_files = list(self.log_dir.glob(f"{base_name}.*{extension}*"))
        rotated_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove oldest rotated files if we have too many
        while len(rotated_files) >= self.max_log_files:
            oldest_file = rotated_files.pop()
            oldest_file.unlink()
        
        # Shift existing rotated files
        for i in range(len(rotated_files) - 1, -1, -1):
            old_number = i + 1
            new_number = old_number + 1
            old_file = rotated_files[i]
            
            # Extract the number from the filename
            if f".{old_number}." in old_file.name:
                new_name = old_file.name.replace(f".{old_number}.", f".{new_number}.")
                new_file = old_file.parent / new_name
                old_file.rename(new_file)
            else:
                # This is the first rotation (no number yet)
                new_name = f"{base_name}.{new_number}{extension}"
                new_file = old_file.parent / new_name
                old_file.rename(new_file)
        
        # Move current log to .1
        rotated_name = f"{base_name}.1{extension}"
        rotated_path = self.log_dir / rotated_name
        
        # Copy current content to rotated file
        shutil.copy2(log_file, rotated_path)
        
        # Compress if enabled
        if self.compression_enabled:
            self.compress_log_file(rotated_path)
        
        # Truncate current log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.truncate()
    
    def compress_log_file(self, log_file: Path):
        """Compress a log file using gzip."""
        compressed_path = log_file.with_suffix(log_file.suffix + '.gz')
        
        try:
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove uncompressed file
            log_file.unlink()
            
        except Exception as e:
            print(f"Failed to compress log file {log_file}: {e}")
    
    def cleanup_old_logs(self):
        """Clean up old log files based on retention policy."""
        print("Starting log cleanup process")
        
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        for log_file in self.log_dir.glob('*.log*'):
            if log_file.stat().st_mtime < cutoff_date.timestamp():
                try:
                    log_file.unlink()
                    deleted_count += 1
                    print(f"Deleted old log file: {log_file.name}")
                except Exception as e:
                    print(f"Failed to delete log file {log_file}: {e}")
        
        print(f"Log cleanup completed. Deleted {deleted_count} files.")
        return deleted_count
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of all log files."""
        summary = {
            'log_directory': str(self.log_dir),
            'total_files': 0,
            'total_size_mb': 0,
            'files': []
        }
        
        for log_file in self.log_dir.glob('*.log*'):
            try:
                stat = log_file.stat()
                file_info = {
                    'name': log_file.name,
                    'path': str(log_file),
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'is_compressed': log_file.suffix == '.gz'
                }
                
                summary['files'].append(file_info)
                summary['total_files'] += 1
                summary['total_size_mb'] += file_info['size_mb']
                
            except Exception as e:
                logger = log_manager.get_logger('main')
                logger.error(f"Failed to get info for log file {log_file}: {e}")
        
        summary['total_size_mb'] = round(summary['total_size_mb'], 2)
        return summary
    
    def analyze_log_content(self, log_file: Path) -> Dict[str, Any]:
        """Analyze the content of a log file."""
        analysis = {
            'file_path': str(log_file),
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'info_count': 0,
            'debug_count': 0,
            'critical_count': 0,
            'date_range': {'start': None, 'end': None},
            'recent_errors': []
        }
        
        if not log_file.exists():
            return analysis
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
            analysis['total_lines'] = len(lines)
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Count log levels
                if ' - ERROR - ' in line:
                    analysis['error_count'] += 1
                elif ' - WARNING - ' in line:
                    analysis['warning_count'] += 1
                elif ' - INFO - ' in line:
                    analysis['info_count'] += 1
                elif ' - DEBUG - ' in line:
                    analysis['debug_count'] += 1
                elif ' - CRITICAL - ' in line:
                    analysis['critical_count'] += 1
                
                # Extract date for date range
                try:
                    # Extract timestamp from log line (assuming format: YYYY-MM-DD HH:MM:SS,mmm)
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                    
                    if analysis['date_range']['start'] is None or timestamp < analysis['date_range']['start']:
                        analysis['date_range']['start'] = timestamp
                    
                    if analysis['date_range']['end'] is None or timestamp > analysis['date_range']['end']:
                        analysis['date_range']['end'] = timestamp
                
                except (ValueError, IndexError):
                    pass
                
                # Collect recent errors
                if ' - ERROR - ' in line and len(analysis['recent_errors']) < 10:
                    analysis['recent_errors'].append(line)
            
            # Convert datetime objects to strings for JSON serialization
            if analysis['date_range']['start']:
                analysis['date_range']['start'] = analysis['date_range']['start'].isoformat()
            if analysis['date_range']['end']:
                analysis['date_range']['end'] = analysis['date_range']['end'].isoformat()
            
        except Exception as e:
            logger = log_manager.get_logger('main')
            logger.error(f"Failed to analyze log file {log_file}: {e}")
        
        return analysis
    
    def generate_log_report(self) -> Dict[str, Any]:
        """Generate a comprehensive log report."""
        logger = log_manager.get_logger('main')
        logger.info("Generating log report")
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': self.get_log_summary(),
            'file_analyses': {}
        }
        
        for log_file in self.log_dir.glob('*.log'):
            try:
                analysis = self.analyze_log_content(log_file)
                report['file_analyses'][log_file.name] = analysis
            except Exception as e:
                logger.error(f"Failed to analyze log file {log_file}: {e}")
        
        return report
    
    def export_log_report(self, output_path: Optional[str] = None):
        """Export log report to a JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.log_dir / f"log_report_{timestamp}.json"
        
        report = self.generate_log_report()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger = log_manager.get_logger('main')
            logger.info(f"Log report exported to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger = log_manager.get_logger('main')
            logger.error(f"Failed to export log report: {e}")
            return None


# Global log rotation manager instance
log_rotation_manager = LogRotationManager()


def rotate_logs():
    """Rotate logs that exceed size limits."""
    return log_rotation_manager.rotate_logs()


def cleanup_old_logs():
    """Clean up old log files."""
    return log_rotation_manager.cleanup_old_logs()


def get_log_summary():
    """Get a summary of all log files."""
    return log_rotation_manager.get_log_summary()


def generate_log_report():
    """Generate a comprehensive log report."""
    return log_rotation_manager.generate_log_report()


def export_log_report(output_path: Optional[str] = None):
    """Export log report to a JSON file."""
    return log_rotation_manager.export_log_report(output_path)
