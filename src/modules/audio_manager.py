"""
Audio manager module for handling audio extraction and playback in video files.
This module provides functionality to extract audio from video files and play it back
synchronized with video playback.
"""

import os
import tempfile
import threading
import time
from typing import Optional, Tuple
import numpy as np
import cv2

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Audio playback will be disabled.")

try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    print("Warning: Wave module not available. Audio playback will be disabled.")

AUDIO_AVAILABLE = PYAUDIO_AVAILABLE and WAVE_AVAILABLE

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("Warning: MoviePy not available. Audio extraction will be disabled.")


class AudioManager:
    """Manages audio extraction and playback for video files."""
    
    def __init__(self):
        """Initialize the audio manager."""
        self.audio_available = AUDIO_AVAILABLE and MOVIEPY_AVAILABLE
        self.current_audio_path = None
        self.audio_thread = None
        self.is_playing = False
        self.volume = 0.7  # Default volume (0.0 to 1.0)
        self.muted = False
        self.sample_rate = 44100
        self.channels = 2
        self.chunk_size = 512  # Reduced chunk size for better sync
        
        # Timing and synchronization variables
        self.playback_start_time = 0.0
        self.playback_start_real_time = 0.0
        self.current_position = 0.0
        self.audio_duration = 0.0
        
        # Initialize PyAudio if available
        if self.audio_available:
            try:
                self.pyaudio = pyaudio.PyAudio()
                self.audio_stream = None
            except Exception as e:
                print(f"Error initializing PyAudio: {e}")
                self.audio_available = False
    
    def extract_audio_from_video(self, video_path: str) -> Optional[str]:
        """
        Extract audio from video file and save as temporary WAV file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Path to the extracted audio file, or None if extraction failed
        """
        if not MOVIEPY_AVAILABLE:
            print("MoviePy not available. Cannot extract audio.")
            return None
        
        # Validate video file exists
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return None
        
        # Check file extension for supported formats
        supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        file_ext = os.path.splitext(video_path)[1].lower()
        if file_ext not in supported_extensions:
            print(f"Unsupported video format: {file_ext}")
            print(f"Supported formats: {', '.join(supported_extensions)}")
            return None
        
        video_clip = None
        audio_path = None
        try:
            # Create temporary file for audio
            temp_dir = tempfile.gettempdir()
            audio_filename = f"audio_{int(time.time())}.wav"
            audio_path = os.path.join(temp_dir, audio_filename)
            
            # Clean up any existing audio file
            if os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            # Extract audio using MoviePy with format-specific handling
            print(f"Extracting audio from {file_ext} file...")
            video_clip = VideoFileClip(video_path)
            
            # Check if video has audio
            if video_clip.audio is None:
                print("Video file has no audio track.")
                return None
            
            # Get audio properties
            audio_duration = video_clip.audio.duration
            audio_fps = video_clip.audio.fps if hasattr(video_clip.audio, 'fps') else 44100
            
            print(f"Audio properties: duration={audio_duration:.2f}s, fps={audio_fps}")
            
            # Extract audio with format-specific parameters
            if file_ext == '.mp4':
                # MP4 files often have better audio quality
                bitrate = '256k'
            elif file_ext == '.avi':
                # AVI files may have lower quality audio
                bitrate = '192k'
            elif file_ext == '.mov':
                # MOV files (QuickTime) usually have good audio
                bitrate = '256k'
            else:
                # Default bitrate for other formats
                bitrate = '192k'
            
            # Extract audio and save as WAV
            video_clip.audio.write_audiofile(
                audio_path,
                fps=audio_fps,
                nbytes=2,  # 16-bit
                codec='pcm_s16le',
                verbose=False,
                logger=None,
                bitrate=bitrate
            )
            
            # Verify the audio file was created and has content
            if not os.path.exists(audio_path):
                print("Failed to create audio file.")
                return None
            
            # Check file size
            file_size = os.path.getsize(audio_path)
            if file_size < 1024:  # Less than 1KB
                print(f"Audio file too small: {file_size} bytes")
                return None
            
            print(f"Audio extracted successfully: {audio_path} ({file_size} bytes)")
            return audio_path
            
        except Exception as e:
            print(f"Error extracting audio: {e}")
            # Clean up any partially created audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            return None
        finally:
            # Always close the video clip
            if video_clip:
                try:
                    video_clip.close()
                except:
                    pass
    
    def play_audio(self, audio_path: str, start_time: float = 0.0) -> bool:
        """
        Play audio file in a separate thread.
        
        Args:
            audio_path: Path to the audio file
            start_time: Start time in seconds
            
        Returns:
            True if playback started successfully, False otherwise
        """
        if not self.audio_available:
            print("Audio playback not available.")
            return False
        
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return False
        
        # Stop any existing playback
        self.stop_audio()
        
        self.current_audio_path = audio_path
        self.playback_start_time = start_time
        self.playback_start_real_time = time.time()
        self.current_position = start_time
        
        # Get audio duration for better timing
        try:
            with wave.open(audio_path, 'rb') as wav_file:
                total_frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                self.audio_duration = total_frames / sample_rate if sample_rate > 0 else 0.0
        except Exception as e:
            print(f"Warning: Could not get audio duration: {e}")
            self.audio_duration = 0.0
        
        print(f"Starting audio playback at {start_time:.3f}s (duration: {self.audio_duration:.3f}s)")
        
        # Start audio playback in separate thread
        self.audio_thread = threading.Thread(
            target=self._play_audio_thread,
            args=(start_time,),
            daemon=True
        )
        self.audio_thread.start()
        
        return True
    
    def _play_audio_thread(self, start_time: float):
        """Thread function for audio playback."""
        wav_file = None
        try:
            # Check if audio file exists
            if not os.path.exists(self.current_audio_path):
                print(f"Audio file not found: {self.current_audio_path}")
                return
            
            # Open the WAV file
            wav_file = wave.open(self.current_audio_path, 'rb')
            
            # Get audio properties
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            chunk_size = self.chunk_size
            
            # Calculate start position
            start_frame = int(start_time * sample_rate)
            total_frames = wav_file.getnframes()
            
            # Validate start position
            if start_frame >= total_frames:
                print(f"Start time {start_time}s is beyond audio duration")
                return
            
            wav_file.setpos(start_frame)
            
            # Open audio stream
            self.audio_stream = self.pyaudio.open(
                format=self.pyaudio.get_format_from_width(wav_file.getsampwidth()),
                channels=channels,
                rate=sample_rate,
                output=True,
                frames_per_buffer=chunk_size
            )
            
            self.is_playing = True
            print(f"Audio playback started at {start_time:.3f}s")
            
            # Read and play audio data
            frames_played = 0
            while self.is_playing:
                data = wav_file.readframes(chunk_size)
                if not data:
                    break  # End of file
                
                # Update current position
                frames_played += len(data) // (wav_file.getsampwidth() * channels)
                self.current_position = start_time + (frames_played / sample_rate)
                
                # Apply volume control
                if not self.muted and self.volume < 1.0:
                    # Convert bytes to numpy array, apply volume, convert back
                    audio_array = np.frombuffer(data, dtype=np.int16)
                    audio_array = (audio_array * self.volume).astype(np.int16)
                    data = audio_array.tobytes()
                
                # Play the chunk
                self.audio_stream.write(data)
            
            print(f"Audio playback completed at {self.current_position:.3f}s")
            
        except Exception as e:
            print(f"Error in audio playback thread: {e}")
        finally:
            # Clean up
            if wav_file:
                try:
                    wav_file.close()
                except:
                    pass
            
            if self.audio_stream:
                try:
                    self.audio_stream.stop_stream()
                    self.audio_stream.close()
                except:
                    pass
                finally:
                    self.audio_stream = None
            
            self.is_playing = False
    
    def stop_audio(self):
        """Stop audio playback."""
        if self.is_playing:
            self.is_playing = False
            print(f"Audio playback stopped at {self.current_position:.3f}s")
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)  # Increased timeout
        self.audio_thread = None
        
        # Reset timing variables
        self.playback_start_time = 0.0
        self.playback_start_real_time = 0.0
        self.current_position = 0.0
    
    def get_current_position(self) -> float:
        """
        Get current audio playback position.
        
        Returns:
            Current position in seconds
        """
        if self.is_playing and self.current_position > 0:
            return self.current_position
        elif self.playback_start_time > 0:
            # If not playing but we have start time, return the start position
            return self.playback_start_time
        else:
            return 0.0
    
    def set_volume(self, volume: float):
        """
        Set audio volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.volume = max(0.0, min(1.0, volume))
    
    def toggle_mute(self):
        """Toggle mute state."""
        self.muted = not self.muted
        return self.muted
    
    def get_audio_info(self, video_path: str) -> Optional[dict]:
        """
        Get audio information from video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with audio information, or None if no audio
        """
        if not MOVIEPY_AVAILABLE:
            return None
        
        try:
            video_clip = VideoFileClip(video_path)
            
            if video_clip.audio is None:
                video_clip.close()
                return None
            
            audio_info = {
                'has_audio': True,
                'duration': video_clip.audio.duration,
                'sample_rate': video_clip.audio.fps,
                'nchannels': video_clip.audio.nchannels
            }
            
            video_clip.close()
            return audio_info
            
        except Exception as e:
            print(f"Error getting audio info: {e}")
            return None
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_audio()
        
        # Terminate PyAudio if available
        if hasattr(self, 'pyaudio') and self.pyaudio is not None:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
        
        # Clean up temporary audio files
        if self.current_audio_path and os.path.exists(self.current_audio_path):
            try:
                os.remove(self.current_audio_path)
                print(f"Cleaned up temporary audio file: {self.current_audio_path}")
            except Exception as e:
                print(f"Error cleaning up audio file: {e}")
        
        # Reset state
        self.current_audio_path = None
        self.audio_available = False


def create_audio_manager() -> AudioManager:
    """Create and return an AudioManager instance."""
    return AudioManager()
