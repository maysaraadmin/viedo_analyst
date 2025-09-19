"""
Log viewer utility for video analysis application.
Provides a command-line interface for viewing, filtering, and analyzing logs.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re

from .config import config
from .logging_manager import log_manager
from .log_rotation import log_rotation_manager
from .log_analyzer import log_analyzer, log_monitor


class LogViewer:
    """Command-line log viewer with filtering and analysis capabilities."""
    
    def __init__(self):
        self.log_dir = Path(config.logs_directory)
        self.colors = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m', # Magenta
            'RESET': '\033[0m'      # Reset
        }
    
    def list_log_files(self) -> List[Dict[str, Any]]:
        """List all available log files with their information."""
        log_files = []
        
        for log_file in self.log_dir.glob('*.log*'):
            try:
                stat = log_file.stat()
                file_info = {
                    'name': log_file.name,
                    'path': str(log_file),
                    'size_bytes': stat.st_size,
                    'size_mb': round(stat.st_size / (1024 * 1024), 2),
                    'last_modified': datetime.fromtimestamp(stat.st_mtime),
                    'is_compressed': log_file.suffix == '.gz'
                }
                log_files.append(file_info)
            except Exception as e:
                print(f"Error getting info for {log_file}: {e}")
        
        return sorted(log_files, key=lambda x: x['last_modified'], reverse=True)
    
    def view_log_file(self, filename: str, lines: int = 50, 
                     level_filter: Optional[str] = None, 
                     search_pattern: Optional[str] = None,
                     follow: bool = False):
        """View a log file with optional filtering."""
        log_file = self.log_dir / filename
        
        if not log_file.exists():
            print(f"Log file not found: {log_file}")
            return
        
        try:
            if follow:
                self._follow_log_file(log_file, level_filter, search_pattern)
            else:
                self._display_log_content(log_file, lines, level_filter, search_pattern)
                
        except Exception as e:
            print(f"Error reading log file: {e}")
    
    def _display_log_content(self, log_file: Path, lines: int, 
                           level_filter: Optional[str], search_pattern: Optional[str]):
        """Display log content with filtering."""
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            all_lines = f.readlines()
        
        # Filter lines
        filtered_lines = []
        for line in all_lines:
            if self._should_include_line(line, level_filter, search_pattern):
                filtered_lines.append(line)
        
        # Show last N lines
        display_lines = filtered_lines[-lines:] if lines > 0 else filtered_lines
        
        for line in display_lines:
            self._print_colored_line(line.strip())
    
    def _follow_log_file(self, log_file: Path, level_filter: Optional[str], 
                        search_pattern: Optional[str]):
        """Follow a log file in real-time."""
        print(f"Following log file: {log_file}")
        print("Press Ctrl+C to stop following...")
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                # Go to end of file
                f.seek(0, 2)
                
                while True:
                    line = f.readline()
                    if line:
                        if self._should_include_line(line, level_filter, search_pattern):
                            self._print_colored_line(line.strip())
                    else:
                        import time
                        time.sleep(0.1)
                        
        except KeyboardInterrupt:
            print("\nStopped following log file.")
        except Exception as e:
            print(f"Error following log file: {e}")
    
    def _should_include_line(self, line: str, level_filter: Optional[str], 
                           search_pattern: Optional[str]) -> bool:
        """Check if a line should be included based on filters."""
        # Level filter
        if level_filter and f' - {level_filter.upper()} - ' not in line:
            return False
        
        # Search pattern
        if search_pattern and search_pattern.lower() not in line.lower():
            return False
        
        return True
    
    def _print_colored_line(self, line: str):
        """Print a line with appropriate color based on log level."""
        for level, color in self.colors.items():
            if level != 'RESET' and f' - {level} - ' in line:
                print(f"{color}{line}{self.colors['RESET']}")
                return
        
        # Default color for lines without recognized log level
        print(line)
    
    def search_logs(self, pattern: str, filename: Optional[str] = None, 
                   context_lines: int = 2) -> List[Dict[str, Any]]:
        """Search for a pattern in log files."""
        results = []
        
        if filename:
            log_files = [self.log_dir / filename]
        else:
            log_files = list(self.log_dir.glob('*.log'))
        
        for log_file in log_files:
            if not log_file.exists():
                continue
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                
                for i, line in enumerate(lines):
                    if pattern.lower() in line.lower():
                        result = {
                            'filename': log_file.name,
                            'line_number': i + 1,
                            'match_line': line.strip(),
                            'context_before': [lines[j].strip() for j in 
                                             range(max(0, i - context_lines), i)],
                            'context_after': [lines[j].strip() for j in 
                                            range(i + 1, min(len(lines), i + context_lines + 1))]
                        }
                        results.append(result)
                        
            except Exception as e:
                print(f"Error searching in {log_file}: {e}")
        
        return results
    
    def show_log_statistics(self, filename: Optional[str] = None, 
                          hours_back: int = 24) -> Dict[str, Any]:
        """Show statistics for log files."""
        if filename:
            log_file = self.log_dir / filename
            if log_file.exists():
                return log_analyzer.analyze_log_file(log_file, hours_back)
            else:
                print(f"Log file not found: {log_file}")
                return {}
        else:
            return log_analyzer.analyze_all_logs(hours_back)
    
    def export_logs(self, filename: str, output_path: str, 
                   format_type: str = 'json', 
                   level_filter: Optional[str] = None,
                   search_pattern: Optional[str] = None):
        """Export log data to a file."""
        log_file = self.log_dir / filename
        
        if not log_file.exists():
            print(f"Log file not found: {log_file}")
            return
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # Filter lines
            filtered_lines = []
            for line in lines:
                if self._should_include_line(line, level_filter, search_pattern):
                    filtered_lines.append(line.strip())
            
            # Export based on format
            if format_type.lower() == 'json':
                export_data = {
                    'source_file': filename,
                    'export_time': datetime.now().isoformat(),
                    'total_lines': len(filtered_lines),
                    'lines': filtered_lines
                }
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format_type.lower() == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'level', 'logger', 'message'])
                    
                    for line in filtered_lines:
                        parts = line.split(' - ', 3)
                        if len(parts) >= 4:
                            writer.writerow(parts)
                        else:
                            writer.writerow(['', '', '', line])
            
            else:  # Plain text
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(filtered_lines))
            
            print(f"Exported {len(filtered_lines)} lines to {output_path}")
            
        except Exception as e:
            print(f"Error exporting logs: {e}")


def main():
    """Main entry point for the log viewer CLI."""
    parser = argparse.ArgumentParser(description='Video Analysis Log Viewer')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available log files')
    
    # View command
    view_parser = subparsers.add_parser('view', help='View a log file')
    view_parser.add_argument('filename', help='Log file name')
    view_parser.add_argument('-n', '--lines', type=int, default=50, 
                           help='Number of lines to show (default: 50)')
    view_parser.add_argument('-l', '--level', 
                           choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                           help='Filter by log level')
    view_parser.add_argument('-s', '--search', help='Search pattern')
    view_parser.add_argument('-f', '--follow', action='store_true',
                           help='Follow log file in real-time')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for patterns in logs')
    search_parser.add_argument('pattern', help='Search pattern')
    search_parser.add_argument('-f', '--filename', help='Specific log file to search')
    search_parser.add_argument('-c', '--context', type=int, default=2,
                             help='Number of context lines (default: 2)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show log statistics')
    stats_parser.add_argument('-f', '--filename', help='Specific log file')
    stats_parser.add_argument('-h', '--hours', type=int, default=24,
                            help='Hours of data to analyze (default: 24)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export log data')
    export_parser.add_argument('filename', help='Log file name')
    export_parser.add_argument('output', help='Output file path')
    export_parser.add_argument('-t', '--type', choices=['json', 'csv', 'txt'],
                             default='json', help='Export format (default: json)')
    export_parser.add_argument('-l', '--level', 
                             choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                             help='Filter by log level')
    export_parser.add_argument('-s', '--search', help='Search pattern')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Start real-time log monitoring')
    monitor_parser.add_argument('-d', '--duration', type=int, default=60,
                              help='Monitoring duration in seconds (default: 60)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    viewer = LogViewer()
    
    try:
        if args.command == 'list':
            log_files = viewer.list_log_files()
            print(f"{'Filename':<25} {'Size (MB)':<10} {'Last Modified':<20} {'Compressed'}")
            print("-" * 70)
            for file_info in log_files:
                compressed = "Yes" if file_info['is_compressed'] else "No"
                print(f"{file_info['name']:<25} {file_info['size_mb']:<10.2f} "
                      f"{file_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S'):<20} {compressed}")
        
        elif args.command == 'view':
            viewer.view_log_file(args.filename, args.lines, args.level, args.search, args.follow)
        
        elif args.command == 'search':
            results = viewer.search_logs(args.pattern, args.filename, args.context)
            print(f"Found {len(results)} matches for pattern: {args.pattern}")
            print("-" * 80)
            
            for result in results:
                print(f"\nFile: {result['filename']}, Line {result['line_number']}")
                print("Context before:")
                for ctx_line in result['context_before']:
                    print(f"  {ctx_line}")
                print(f"Match: {result['match_line']}")
                print("Context after:")
                for ctx_line in result['context_after']:
                    print(f"  {ctx_line}")
                print("-" * 40)
        
        elif args.command == 'stats':
            stats = viewer.show_log_statistics(args.filename, args.hours)
            if stats:
                if args.filename:
                    print(f"Statistics for {args.filename}:")
                    print(f"Total lines: {stats['total_lines']}")
                    print(f"Errors: {stats['error_count']}")
                    print(f"Warnings: {stats['warning_count']}")
                    if stats['alerts']:
                        print("Alerts:")
                        for alert in stats['alerts']:
                            print(f"  - {alert}")
                else:
                    print(f"Overall statistics (last {args.hours} hours):")
                    print(f"Total files: {stats['summary']['total_files']}")
                    print(f"Total errors: {stats['summary']['total_errors']}")
                    print(f"Total warnings: {stats['summary']['total_warnings']}")
                    if stats['summary']['all_alerts']:
                        print("All alerts:")
                        for alert in stats['summary']['all_alerts']:
                            print(f"  - {alert}")
        
        elif args.command == 'export':
            viewer.export_logs(args.filename, args.output, args.type, args.level, args.search)
        
        elif args.command == 'monitor':
            print(f"Starting log monitoring for {args.duration} seconds...")
            log_monitor.start_monitoring()
            
            import time
            start_time = time.time()
            while time.time() - start_time < args.duration:
                alerts = log_monitor.get_recent_alerts()
                if alerts:
                    print(f"\n{len(alerts)} new alerts:")
                    for alert in alerts:
                        print(f"  [{alert['timestamp']}] {alert['type']}: {alert['message']}")
                time.sleep(5)
            
            log_monitor.stop_monitoring()
            print("Monitoring stopped.")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
