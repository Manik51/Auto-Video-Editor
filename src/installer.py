"""
installer.py
Environment initialization and smart dependency installer for AutoVideoEditor.
Optimized for Google Colab and local environments with GPU acceleration.
"""

import os
import sys
import subprocess
import importlib.util
import urllib.request
import shutil
from typing import Dict, Any, List


REQUIRED_PACKAGES = {
    "moviepy": "moviepy==1.0.3",
    "cv2": "opencv-python>=4.8.0",
    "ffmpeg": "ffmpeg-python>=0.2.0",
    "scenedetect": "scenedetect[opencv]>=0.6.2",
    "whisper": "openai-whisper",
    "librosa": "librosa>=0.10.1",
    "soundfile": "soundfile>=0.12.1",
    "noisereduce": "noisereduce>=3.0.0",
    "gfpgan": "gfpgan>=1.3.8",
    "rembg": "rembg>=2.0.50",
    "gradio": "gradio>=4.20.0",
    "PIL": "Pillow>=10.0.0",
    "numpy": "numpy>=1.24.0",
    "scipy": "scipy>=1.11.0",
    "tqdm": "tqdm>=4.66.0",
}

FONT_URLS = {
    "Montserrat-Bold.ttf": "https://github.com/googlefonts/montserrat/raw/master/fonts/ttf/Montserrat-Bold.ttf",
    "Roboto-Bold.ttf": "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf",
}

GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


def is_package_installed(module_name: str) -> bool:
    """Check if a Python package module can be imported without importing it fully."""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ValueError, AttributeError):
        return False


def install_package(pip_spec: str) -> bool:
    """Install a specific pip package cleanly."""
    try:
        print(f"📦 Installing {pip_spec}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-warn-script-location", pip_spec],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Failed to install {pip_spec}: {e}")
        return False


def check_and_install_dependencies(force: bool = False) -> Dict[str, bool]:
    """
    Check all required dependencies and install missing ones.
    Uses smart caching to avoid redundant installations on notebook restarts.
    """
    status = {}
    print("=" * 60)
    print("🔍 AutoVideoEditor: Checking Dependencies & Environment...")
    print("=" * 60)

    for mod_name, pip_spec in REQUIRED_PACKAGES.items():
        if not force and is_package_installed(mod_name):
            status[mod_name] = True
            print(f"  ✅ {mod_name:<16} : Already installed")
        else:
            success = install_package(pip_spec)
            status[mod_name] = success
            if success:
                print(f"  ✅ {mod_name:<16} : Successfully installed")
            else:
                print(f"  ❌ {mod_name:<16} : Installation failed")

    return status


def check_system_resources() -> Dict[str, Any]:
    """
    Inspect GPU, CPU, RAM, and FFmpeg configuration.
    """
    info = {
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
        "ffmpeg_available": False,
        "ffmpeg_version": None,
    }

    # Check GPU via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_gb"] = round(vram_bytes / (1024 ** 3), 2)
            print(f"🚀 CUDA GPU Ready: {info['gpu_name']} ({info['gpu_memory_gb']} GB VRAM)")
        else:
            print("⚠️ No CUDA GPU detected. System will run in CPU mode (processing will be slower).")
    except Exception as e:
        print(f"ℹ️ PyTorch check notice: {e}")

    # Check FFmpeg binary
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        info["ffmpeg_available"] = True
        try:
            res = subprocess.run([ffmpeg_bin, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            info["ffmpeg_version"] = res.stdout.splitlines()[0] if res.stdout else "Available"
            print(f"🎬 FFmpeg Ready: {info['ffmpeg_version']}")
        except Exception:
            info["ffmpeg_version"] = "Available"
    else:
        print("⚠️ Warning: FFmpeg executable not found in system PATH. Video export may fail.")

    return info


def download_fonts(fonts_dir: str = None) -> None:
    """Download required fonts if not already present."""
    if fonts_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fonts_dir = os.path.join(base_dir, "fonts")
    os.makedirs(fonts_dir, exist_ok=True)

    headers = {"User-Agent": "AutoVideoEditor-Installer/1.0"}
    for font_name, url in FONT_URLS.items():
        dest_path = os.path.join(fonts_dir, font_name)
        if not os.path.exists(dest_path) or os.path.getsize(dest_path) < 1000:
            print(f"📥 Downloading font {font_name}...")
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
                    f.write(resp.read())
                print(f"  ✅ Saved {font_name} ({os.path.getsize(dest_path)} bytes)")
            except Exception as e:
                print(f"  ⚠️ Could not download font {font_name}: {e}")
        else:
            print(f"  ✅ Font {font_name} is already available.")


def download_models(models_dir: str = None) -> None:
    """Pre-download GFPGAN weights and initialize Whisper cache."""
    if models_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Download GFPGAN weights if missing
    gfpgan_dest = os.path.join(models_dir, "GFPGANv1.4.pth")
    if not os.path.exists(gfpgan_dest) or os.path.getsize(gfpgan_dest) < 10000000:
        print(f"📥 Downloading GFPGAN v1.4 model weights (approx. 348MB)...")
        try:
            req = urllib.request.Request(GFPGAN_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(gfpgan_dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            print(f"  ✅ GFPGAN v1.4 weights saved ({os.path.getsize(gfpgan_dest)} bytes)")
        except Exception as e:
            print(f"  ⚠️ Notice: Could not pre-download GFPGAN weights: {e}. Will download on first use.")
    else:
        print("  ✅ GFPGAN v1.4 weights already cached.")

    # Pre-cache Whisper base model
    try:
        import whisper
        print("📥 Pre-loading Whisper base model...")
        whisper.load_model("base")
        print("  ✅ Whisper base model loaded successfully.")
    except Exception as e:
        print(f"  ℹ️ Whisper pre-load notice: {e}")


def setup_environment() -> Dict[str, Any]:
    """Master environment setup orchestrator."""
    check_and_install_dependencies()
    sys_info = check_system_resources()
    download_fonts()
    return sys_info


if __name__ == "__main__":
    setup_environment()
