"""
AutoVideoEditor Package
Production-grade automated AI video editing framework for Google Colab and local environments.
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

__version__ = "1.0.0"
__author__ = "AutoVideoEditor Team"

from .installer import setup_environment, check_system_resources
from .scene_detector import SceneDetector
from .audio_analyzer import AudioAnalyzer
from .silence_remover import SilenceRemover
from .color_grader import ColorGrader
from .caption_generator import CaptionGenerator
from .transition_engine import TransitionEngine
from .effects_engine import EffectsEngine
from .face_enhancer import FaceEnhancer
from .bg_remover import BackgroundRemover
from .audio_enhancer import AudioEnhancer
from .renderer import VideoRenderer
from .pipeline import VideoPipeline

__all__ = [
    "setup_environment",
    "check_system_resources",
    "SceneDetector",
    "AudioAnalyzer",
    "SilenceRemover",
    "ColorGrader",
    "CaptionGenerator",
    "TransitionEngine",
    "EffectsEngine",
    "FaceEnhancer",
    "BackgroundRemover",
    "AudioEnhancer",
    "VideoRenderer",
    "VideoPipeline",
]
