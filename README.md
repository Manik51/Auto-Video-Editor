# 🎬 AutoVideoEditor AI
### Automated AI-Powered Video Editing Studio (CapCut-Level Autonomous Editing)

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manik51/Auto-Video-Editor/blob/main/main.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab%20%7C%20Windows%20%7C%20Linux-orange.svg)](https://colab.research.google.com)
[![CUDA Acceleration](https://img.shields.io/badge/CUDA-Enabled%20(T4%20%2F%20V100%20%2F%20A100)-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![FFmpeg](https://img.shields.io/badge/FFmpeg-Hardware%20Accelerated-black.svg?logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <b>Transform raw, unedited footage into fully polished, professional-grade videos automatically.</b><br>
  Zero manual timeline trimming required. Built for creators, vloggers, podcasters, and filmmakers.
</p>

</div>

---

## ⚡ 1-Click Quickstart on Google Colab

The easiest and fastest way to use AutoVideoEditor is directly inside **Google Colab** with free GPU acceleration:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Manik51/Auto-Video-Editor/blob/main/main.ipynb)

### 3 Steps to Run in Google Colab:

1. **Open the Notebook**: Click the **"Open In Colab"** badge above, or open `main.ipynb` in [Google Colab](https://colab.research.google.com).
2. **Enable GPU**:
   - Go to top menu: `Runtime` ➔ `Change runtime type`
   - Select **T4 GPU** ➔ Click **Save**
3. **Run All**:
   - Press **`Ctrl + F9`** (or go to `Runtime` ➔ `Run all`).
   - When Cell 6 finishes, click the public web link:  
     `👉 Running on public URL: https://xxxx.gradio.live`
   - Upload your raw video, select a style preset, and click **🚀 Generate Edited Video**!

---

## 🌟 Key Features

| Feature | Powered By | Description |
| :--- | :--- | :--- |
| **Smart Auto-Cutting** | `PySceneDetect` | Detects scene boundaries, removes static/boring clips, scores remaining scenes by motion, face presence, and audio energy, keeping the top 70% highest quality footage. |
| **Silence & Dead-Air Removal** | `Librosa` | Detects pauses below -40dB (>0.5s), applies frame-accurate cuts with 0.1s speech safety padding so speech starts/ends naturally without abrupt consonant clipping. |
| **Beat-Sync Editing** | `Librosa Beat Tracking` | Automatically detects BPM and musical bar timestamps to snap cuts and zoom punch effects to the rhythm of your soundtrack. |
| **Cinematic Color Grading** | Hardware 3D LUTs (`.cube`) | Applies cinematic LUTs (Orange & Teal, Warm Golden, Film Noir, Cold Blue) with auto white balance, CLAHE dynamic range leveling, +15% saturation, vignette, and 35mm film grain. |
| **AI Auto Subtitles** | `OpenAI Whisper` | Generates word-level timestamps and burns high-retention subtitles directly into video frames (Montserrat Bold, white text, black outline, 10% bottom margin). |
| **Smooth Transitions** | `MoviePy / OpenCV` | Seamless non-repeating transitions between cuts: Cross Dissolve (0.3s), Fade to Black (0.3s), Zoom Punch (0.15s), Glitch (3 frames RGB split), and Whip Pan Blur (0.2s). |
| **Dynamic Camera Effects** | Motion Engine | Ken Burns slow zoom (1.0x -> 1.05x) on static footage, 0.5x speed ramp slow-mo, audio peak zoom punches, and video stabilization. |
| **9:16 Auto-Reframing** | Face Centering | Automatically crops widescreen 16:9 landscape footage into vertical 9:16 portrait for Instagram Reels / TikTok / YouTube Shorts using continuous face tracking. |
| **Audio Post-Production** | `Noisereduce & Scipy` | Spectral background noise removal, 2kHz-4kHz speech presence EQ boost, -14 LUFS broadcast loudness normalization, and -18dB background music ducking. |
| **AI Face Enhancement** | `GFPGAN v1.4` | High-definition facial restoration on detected faces with smooth temporal blending (GPU accelerated). |
| **High-Speed FFmpeg Export** | `FFmpeg Subprocess` | Broadcast-grade encoding (CRF 18, slow preset, AAC 320kbps 48kHz, yuv420p) with chunked memory management for long videos (>10 min). |

---

## 📁 Repository Structure

```
AutoVideoEditor/
├── main.ipynb                  # Master Google Colab notebook (1-Click runner)
├── README.md                   # Full documentation & setup guide
├── requirements.txt            # Python dependencies
├── START_ON_WINDOWS.bat        # 1-Click Windows desktop launcher
├── src/
│   ├── __init__.py             # Package exports
│   ├── installer.py            # Environment checker & model downloader
│   ├── scene_detector.py       # PySceneDetect scene cutting & quality scoring
│   ├── audio_analyzer.py       # Librosa beat tracking & silence detection
│   ├── silence_remover.py      # Dead-air trimmer with 0.1s padding
│   ├── color_grader.py         # 3D LUT engine, auto WB, CLAHE, vignette, film grain
│   ├── caption_generator.py    # Whisper AI subtitles & ASS/SRT burning
│   ├── transition_engine.py    # Non-repeating cinematic transitions
│   ├── effects_engine.py       # Ken Burns, speed ramp, 9:16 auto-reframe
│   ├── face_enhancer.py        # GFPGAN v1.4 facial restoration
│   ├── bg_remover.py           # Background removal & bokeh blur
│   ├── audio_enhancer.py       # Denoise, EQ boost, -14 LUFS normalization, BGM ducking
│   ├── renderer.py             # FFmpeg multi-pass export engine
│   └── pipeline.py             # Master fault-tolerant controller
├── presets/
│   ├── youtube_vlog.json       # 16:9, warm golden LUT, vlog pacing
│   ├── instagram_reels.json    # 9:16 auto-reframe, fast cuts, 56px font
│   ├── cinematic_film.json     # Orange & teal LUT, cross dissolve, film grain
│   ├── podcast.json            # Talking head focus, strict silence removal, voice EQ boost
│   └── music_video.json        # Librosa beat-sync cuts, drop zoom punch, cold blue LUT
├── luts/
│   ├── cinematic_orange_teal.cube
│   ├── film_noir.cube
│   ├── warm_golden.cube
│   └── cold_blue.cube
├── fonts/
│   ├── Montserrat-Bold.ttf
│   └── Roboto-Bold.ttf
└── ui/
    └── gradio_app.py           # Web UI running inside Colab or local browser
```

---

## 💻 Local Installation (Windows, macOS, Linux)

If you prefer to run the editor locally on your personal computer:

### Option A: 1-Click Launcher (Windows)
Double-click **`START_ON_WINDOWS.bat`** in the root folder. It will launch the interface and open your browser to `http://localhost:7860`.

### Option B: Command Line (All Platforms)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manik51/Auto-Video-Editor.git
   cd Auto-Video-Editor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Web Studio:**
   ```bash
   python ui/gradio_app.py
   ```
   Open your browser at `http://localhost:7860`.

---

## 🎛️ Editing Style Presets

AutoVideoEditor includes 5 tailored presets out of the box:

| Preset | Aspect Ratio | LUT Color Grade | Target Style |
| :--- | :--- | :--- | :--- |
| **YouTube Vlog** | 16:9 (1920x1080) | Warm Golden | Natural warmth, conversational pacing, -14 LUFS audio, subtitles. |
| **Instagram Reels** | 9:16 (1080x1920) | Orange & Teal | Face-centered auto-reframe, punchy 56px captions, high-energy cuts. |
| **Cinematic Film** | 16:9 (1920x1080) | Orange & Teal | S-curve contrast, 35mm film grain, vignette, cross-dissolve transitions. |
| **Podcast** | 16:9 (1920x1080) | Warm Golden | Aggressive dead-air removal, 2kHz-4kHz vocal presence EQ, denoise. |
| **Music Video** | 16:9 (1920x1080) | Cold Blue | Rhythmic cuts snapped to music beats, drop zoom punches, whip pan blurs. |

---

## 🐍 Python Batch / Script API

You can integrate AutoVideoEditor directly into automated backend workflows or video rendering pipelines:

```python
from src.pipeline import VideoPipeline

# Initialize the automated pipeline
pipeline = VideoPipeline(
    input_path="raw_footage.mp4",
    preset="YouTube Vlog",  # or "Instagram Reels", "Cinematic Film", etc.
    output_path="output/edited_master.mp4",
    options={
        "auto_captions": True,
        "color_grade": True,
        "remove_silence": True,
        "face_enhance": False,
        "beat_sync": False,
        "bgm_audio_path": "background_music.mp3",  # Optional
    }
)

# Run end-to-end editing
results = pipeline.run()

print(f"✅ Success: {results['success']}")
print(f"🎬 Output Video: {results['output_path']}")
print(f"⏱️ Duration: {results['original_duration']}s -> {results['final_duration']}s")
```

---

## ⚙️ Hardware Requirements

- **Google Colab (Recommended):** Free T4 GPU runtime.
- **Local GPU (Optional):** NVIDIA GPU with 4GB+ VRAM (for CUDA acceleration).
- **Local CPU:** Multi-core Intel/AMD processor (runs on CPU automatically if CUDA is not available).
- **FFmpeg:** Pre-installed on Google Colab and included in standard system packages.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details. Open-source and free for personal and commercial content creation.

---

<div align="center">
  <b>Built with ❤️ for content creators and video editors worldwide.</b><br>
  ⭐ Star this repo if you find it helpful!
</div>
