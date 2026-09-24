"""Local CPU-only real-time voice agent for IT staffing.

Pipeline:  Mic -> Silero VAD -> distil-Whisper -> Qwen2.5-1.5B-Q4_K_M -> Piper -> Speaker
"""

__version__ = "1.0.0"
