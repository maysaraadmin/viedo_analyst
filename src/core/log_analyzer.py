"""
Log analysis and monitoring utilities for video analysis application.
Provides real-time log monitoring, analysis, and alerting capabilities.
"""

import re
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
import queue

from .config import config
from .logging_manager import log_manager
from .log_rotation import log_rotation_manager


class LogAnalyzer:
    """Analyzes log files for patterns, errors, and performance metrics."""
    
    def __init__(self):
        self.log_dir = Path(config.logs_directory)
        self.alert_thresholds = {
            'error_count_per_hour': 10,
            'warning_count_per_hour': 50,
            'response_time_ms': 5000,
            'memory_usage_mb': 1000
        }
        self.patterns = {
            'error_patterns': [
                r'ERROR - .*',
                r'Exception: .*',
                r'Traceback.*',
                r'Failed to.*',
                r'Error.*'
            ],
            'warning_patterns': [
                r'WARNING - .*',
                r'DeprecationWarning.*',
                r'UserWarning.*'
            ],
            'performance_patterns': [
                r'Performance metric:.*duration_ms":\s*(\d+)',
                r'took\s*(\d+(?:\.\d+)?)\s*seconds?',
                r'completed in\s*(\d+(?:\.\d+)?)\s*ms'
            ],
            'memory_patterns': [
                r'memory usage[:\s]*(\d+(?:\.\d+)?)\s*MB',
                r'Memory[:\s]*(\d+(?:\.\d+)?)\s*MB',
                r'RAM[:\s]*(\d+(?:\.\d+)?)\s*MB'
            ]
        }
        
    def analyze_log_file(self, log_file_path: Path, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze a specific log file for patterns and metrics."""
        analysis = {
            'file_path': str(log_file_path),
            'analysis_time': datetime.now().isoformat(),
            'time_range_hours': hours_back,
            'total_lines': 0,
            'error_count': 0,
            'warning_count': 0,
            'performance_metrics': [],
            'memory_usage': [],
            'error_patterns': defaultdict(int),
            'recent_errors': [],
            'hourly_stats': defaultdict(lambda: {'errors': 0, 'warnings': 0, 'total': 0}),
            'alerts': []
        }
        
        if not log_file_path.exists():
            analysis['alerts'].append(f"Log file not found: {log_file_path}")
            return analysis
        
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        try:
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    analysis['total_lines'] += 1
                    
                    # Extract timestamp
                    timestamp = self._extract_timestamp(line)
                    if timestamp and timestamp < cutoff_time:
                        continue
                    
                    # Analyze line for patterns
                    self._analyze_line(line, analysis, timestamp)
            
            # Generate alerts based on analysis
            analysis['alerts'].extend(self._generate_alerts(analysis))
            
        except Exception as e:
            analysis['alerts'].append(f"Failed to analyze log file: {e}")
        
        return analysis
    
    def _extract_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from log line."""
        try:
            # Try different timestamp formats
            timestamp_formats = [
                '%Y-%m-%d %H:%M:%S,%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d %H:%M:%S.%f'
            ]
            
            timestamp_str = line.split(' - ')[0]
            for fmt in timestamp_formats:
                try:
                    return datetime.strptime(timestamp_str, fmt)
                except ValueError:
                    continue
        except (IndexError, ValueError):
            pass
        
        return None
    
    def _analyze_line(self, line: str, analysis: Dict[str, Any], timestamp: Optional[datetime]):
        """Analyze a single log line."""
        # Check for errors
        if ' - ERROR - ' in line:
            analysis['error_count'] += 1
            analysis['recent_errors'].append(line)
            if len(analysis['recent_errors']) > 20:
                analysis['recent_errors'].pop(0)
        
        # Check for warnings
        elif ' - WARNING - ' in line:
            analysis['warning_count'] += 1
        
        # Check for error patterns
        for pattern in self.patterns['error_patterns']:
            if re.search(pattern, line, re.IGNORECASE):
                analysis['error_patterns'][pattern] += 1
        
        # Extract performance metrics
        perf_matches = re.findall(self.patterns['performance_patterns'][0], line)
        for match in perf_matches:
            try:
                duration_ms = float(match)
                analysis['performance_metrics'].append({
                    'timestamp': timestamp.isoformat() if timestamp else None,
                    'duration_ms': duration_ms
                })
            except ValueError:
                pass
        
        # Extract memory usage
        memory_matches = re.findall(self.patterns['memory_patterns'][0], line)
        for match in memory_matches:
            try:
                memory_mb = float(match)
                analysis['memory_usage'].append({
                    'timestamp': timestamp.isoformat() if timestamp else None,
                    'memory_mb': memory_mb
                })
            except ValueError:
                pass
        
        # Update hourly statistics
        if timestamp:
            hour_key = timestamp.strftime('%Y-%m-%d %H:00')
            if ' - ERROR - ' in line:
                analysis['hourly_stats'][hour_key]['errors'] += 1
            elif ' - WARNING - ' in line:
                analysis['hourly_stats'][hour_key]['warnings'] += 1
            analysis['hourly_stats'][hour_key]['total'] += 1
    
    def _generate_alerts(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate alerts based on analysis results."""
        alerts = []
        
        # Check error rate
        if analysis['error_count'] > self.alert_thresholds['error_count_per_hour']:
            alerts.append(f"High error rate: {analysis['error_count']} errors in last {analysis['time_range_hours']} hours")
        
        # Check warning rate
        if analysis['warning_count'] > self.alert_thresholds['warning_count_per_hour']:
            alerts.append(f"High warning rate: {analysis['warning_count']} warnings in last {analysis['time_range_hours']} hours")
        
        # Check performance metrics
        if analysis['performance_metrics']:
            avg_response_time = sum(m['duration_ms'] for m in analysis['performance_metrics']) / len(analysis['performance_metrics'])
            if avg_response_time > self.alert_thresholds['response_time_ms']:
                alerts.append(f"High average response time: {avg_response_time:.2f}ms")
        
        # Check memory usage
        if analysis['memory_usage']:
            avg_memory = sum(m['memory_mb'] for m in analysis['memory_usage']) / len(analysis['memory_usage'])
            if avg_memory > self.alert_thresholds['memory_usage_mb']:
                alerts.append(f"High average memory usage: {avg_memory:.2f}MB")
        
        return alerts
    
    def analyze_all_logs(self, hours_back: int = 24) -> Dict[str, Any]:
        """Analyze all log files in the log directory."""
        comprehensive_analysis = {
            'analysis_time': datetime.now().isoformat(),
            'time_range_hours': hours_back,
            'file_analyses': {},
            'summary': {
                'total_errors': 0,
                'total_warnings': 0,
                'total_files': 0,
                'all_alerts': []
            }
        }
        
        for log_file in self.log_dir.glob('*.log'):
            try:
                analysis = self.analyze_log_file(log_file, hours_back)
                comprehensive_analysis['file_analyses'][log_file.name] = analysis
                comprehensive_analysis['summary']['total_errors'] += analysis['error_count']
                comprehensive_analysis['summary']['total_warnings'] += analysis['warning_count']
                comprehensive_analysis['summary']['total_files'] += 1
                comprehensive_analysis['summary']['all_alerts'].extend(analysis['alerts'])
            except Exception as e:
                comprehensive_analysis['summary']['all_alerts'].append(f"Failed to analyze {log_file}: {e}")
        
        return comprehensive_analysis


class LogMonitor:
    """Real-time log monitoring and alerting system."""
    
    def __init__(self):
        self.log_dir = Path(config.logs_directory)
        self.running = False
        self.monitor_thread = None
        self.alert_callbacks = []
        self.log_positions = {}
        self.check_interval = 5  # seconds
        self.alert_queue = queue.Queue()
        
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Add a callback function to be called when an alert is triggered."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start real-time log monitoring."""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_logs, daemon=True)
        self.monitor_thread.start()
        
        logger = log_manager.get_logger('main')
        logger.info("Log monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time log monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger = log_manager.get_logger('main')
        logger.info("Log monitoring stopped")
    
    def _monitor_logs(self):
        """Monitor log files for new entries and generate alerts."""
        while self.running:
            try:
                for log_file in self.log_dir.glob('*.log'):
                    self._check_log_file(log_file)
                
                # Process alert queue
                self._process_alerts()
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger = log_manager.get_logger('main')
                logger.error(f"Error in log monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_log_file(self, log_file: Path):
        """Check a single log file for new entries."""
        try:
            if not log_file.exists():
                return
            
            # Get current file position
            current_pos = self.log_positions.get(str(log_file), 0)
            
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(current_pos)
                new_lines = f.readlines()
                self.log_positions[str(log_file)] = f.tell()
            
            # Analyze new lines
            for line in new_lines:
                self._analyze_log_line(line.strip(), log_file.name)
                
        except Exception as e:
            logger = log_manager.get_logger('main')
            logger.error(f"Error checking log file {log_file}: {e}")
    
    def _analyze_log_line(self, line: str, filename: str):
        """Analyze a single log line for alerts."""
        if not line:
            return
        
        # Check for critical errors
        if ' - CRITICAL - ' in line or ' - ERROR - ' in line:
            alert_data = {
                'type': 'error',
                'severity': 'high' if 'CRITICAL' in line else 'medium',
                'message': line,
                'filename': filename,
                'timestamp': datetime.now().isoformat()
            }
            self.alert_queue.put(alert_data)
        
        # Check for specific error patterns
        error_patterns = [
            r'Exception:',
            r'Traceback',
            r'Failed to',
            r'Connection error',
            r'Timeout',
            r'Out of memory'
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                alert_data = {
                    'type': 'pattern_match',
                    'severity': 'medium',
                    'pattern': pattern,
                    'message': line,
                    'filename': filename,
                    'timestamp': datetime.now().isoformat()
                }
                self.alert_queue.put(alert_data)
                break
    
    def _process_alerts(self):
        """Process alerts from the queue and call callbacks."""
        while not self.alert_queue.empty():
            try:
                alert_data = self.alert_queue.get_nowait()
                
                # Log the alert
                logger = log_manager.get_logger('main')
                logger.warning(f"ALERT: {alert_data['type']} - {alert_data['message']}")
                
                # Call registered callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert_data['type'], alert_data)
                    except Exception as e:
                        logger.error(f"Error in alert callback: {e}")
                
            except queue.Empty:
                break
            except Exception as e:
                logger = log_manager.get_logger('main')
                logger.error(f"Error processing alert: {e}")
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts from the queue."""
        alerts = []
        while not self.alert_queue.empty() and len(alerts) < count:
            try:
                alerts.append(self.alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts


# Global instances
log_analyzer = LogAnalyzer()
log_monitor = LogMonitor()


def analyze_log_file(log_file_path: str, hours_back: int = 24) -> Dict[str, Any]:
    """Analyze a specific log file."""
    return log_analyzer.analyze_log_file(Path(log_file_path), hours_back)


def analyze_all_logs(hours_back: int = 24) -> Dict[str, Any]:
    """Analyze all log files."""
    return log_analyzer.analyze_all_logs(hours_back)


def start_log_monitoring():
    """Start real-time log monitoring."""
    log_monitor.start_monitoring()


def stop_log_monitoring():
    """Stop real-time log monitoring."""
    log_monitor.stop_monitoring()


def add_alert_callback(callback: Callable[[str, Dict[str, Any]], None]):
    """Add an alert callback."""
    log_monitor.add_alert_callback(callback)


def get_recent_alerts(count: int = 50) -> List[Dict[str, Any]]:
    """Get recent alerts."""
    return log_monitor.get_recent_alerts(count)
