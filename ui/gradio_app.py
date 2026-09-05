"""
gradio_app.py
Web UI for AutoVideoEditor running inside Google Colab and local environments.
Provides video and music upload, preset selection, feature toggles, real-time progress bar,
and final video preview with download link.
"""

import os
import sys
import time
import tempfile
import gradio as gr

# Ensure parent directory is in Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import VideoPipeline


CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    max-width: 1200px !important;
    margin: auto;
}
.header-box {
    background: linear-gradient(135deg, #1E1E2F 0%, #2A2A44 100%);
    color: white;
    padding: 24px;
    border-radius: 12px;
    margin-bottom: 20px;
    border: 1px solid #3E3E5E;
    text-align: center;
}
.stat-card {
    background-color: #262738;
    padding: 14px;
    border-radius: 8px;
    border: 1px solid #3E4058;
    text-align: center;
}
.stat-title {
    font-size: 13px;
    color: #A0A3BD;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: #4E88FF;
    margin-top: 4px;
}
"""


def process_video(
    video_file,
    preset_name,
    music_file,
    color_look,
    auto_captions,
    color_grade,
    remove_silence,
    face_enhance,
    beat_sync,
    output_format,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Handles user submission, bridges Gradio progress bar, and runs VideoPipeline.
    """
    if not video_file:
        return (
            None,
            None,
            "⚠️ Error: Please upload a raw video file first.",
            "0.0s",
            "0.0s",
            "0",
            "0",
        )

    # Resolve paths
    video_path = video_file if isinstance(video_file, str) else video_file.name
    bgm_path = None
    if music_file:
        bgm_path = music_file if isinstance(music_file, str) else music_file.name

    # Set up resolution from format selector
    res_map = {
        "YouTube 1080p": [1920, 1080],
        "Reels 9:16": [1080, 1920],
        "4K (3840x2160)": [3840, 2160],
    }
    chosen_res = res_map.get(output_format, [1920, 1080])

    # Pipeline options
    options = {
        "auto_captions": auto_captions,
        "color_grade": color_grade,
        "color_look": color_look,
        "remove_silence": remove_silence,
        "face_enhance": face_enhance,
        "beat_sync": beat_sync,
        "bgm_audio_path": bgm_path,
        "resolution": chosen_res,
    }

    # Progress callback bridge
    def ui_progress_callback(percent: float, message: str):
        fraction = max(0.0, min(1.0, percent / 100.0))
        progress(fraction, desc=message)

    pipeline = VideoPipeline(
        input_path=video_path,
        preset=preset_name,
        options=options,
        progress_callback=ui_progress_callback,
    )

    try:
        results = pipeline.run()
        out_path = results.get("output_path")
        orig_dur = f"{results.get('original_duration', 0):.1f}s"
        final_dur = f"{results.get('final_duration', 0):.1f}s"
        scenes = str(results.get("scenes_detected", 0))
        silences = str(results.get("silences_removed", 0))
        log_text = results.get("log", "")

        return (
            out_path,
            out_path,
            log_text,
            orig_dur,
            final_dur,
            scenes,
            silences,
        )
    except Exception as e:
        import traceback
        err_msg = f"❌ Pipeline Failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        return (
            None,
            None,
            err_msg,
            "0.0s",
            "0.0s",
            "0",
            "0",
        )


def create_gradio_ui():
    """Build Gradio UI Blocks application."""
    try:
        demo = gr.Blocks(title="AutoVideoEditor AI")
    except Exception:
        demo = gr.Blocks()
    with demo:
        with gr.Row():
            gr.HTML("""
            <div class="header-box">
                <h1 style="margin: 0; font-size: 32px; font-weight: 800;">🎬 AutoVideoEditor AI</h1>
                <p style="margin: 8px 0 0 0; color: #D1D5DB; font-size: 15px;">
                    Professional CapCut-level automated video editing in Google Colab.
                    Raw footage in → Fully finished cinematic video out.
                </p>
            </div>
            """)

        with gr.Row():
            # Left Column: Inputs & Configuration
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Upload Media")
                video_input = gr.Video(
                    label="Upload Raw Video (MP4, MOV, AVI, MKV)",
                    sources=["upload"],
                )
                music_input = gr.Audio(
                    label="Background Music (Optional MP3 / WAV)",
                    type="filepath",
                    sources=["upload"],
                )

                gr.Markdown("### ⚙️ 2. Editing Settings")
                preset_dropdown = gr.Dropdown(
                    choices=[
                        "YouTube Vlog",
                        "Instagram Reels",
                        "Cinematic Film",
                        "Podcast",
                        "Music Video",
                    ],
                    value="YouTube Vlog",
                    label="Editing Style Preset",
                )

                format_dropdown = gr.Dropdown(
                    choices=["YouTube 1080p", "Reels 9:16", "4K (3840x2160)"],
                    value="YouTube 1080p",
                    label="Export Output Format",
                )

                color_look_dropdown = gr.Dropdown(
                    choices=[
                        "Clean & Crisp (Studio HD)",
                        "Vibrant Pop (Social Media / Reels)",
                        "Cinematic Film (Hollywood Style)",
                        "Warm Golden Hour",
                        "Original / Natural (Untouched)",
                    ],
                    value="Clean & Crisp (Studio HD)",
                    label="🎨 Color Grading & Look Style",
                )

                with gr.Row():
                    chk_captions = gr.Checkbox(label="Auto Captions (Whisper)", value=True)
                    chk_color = gr.Checkbox(label="Color Grade & Enhance", value=True)
                    chk_silence = gr.Checkbox(label="Remove Dead-Air Silence", value=True)

                with gr.Row():
                    chk_face = gr.Checkbox(label="Face Enhance (GFPGAN GPU)", value=False)
                    chk_beat = gr.Checkbox(label="Beat-Sync Cuts", value=False)

                process_btn = gr.Button("🚀 Generate Edited Video", variant="primary", size="lg")

            # Right Column: Outputs & Statistics
            with gr.Column(scale=6):
                gr.Markdown("### 📺 3. Processed Output")
                output_video_player = gr.Video(label="Preview Edited Video", interactive=False)
                download_file = gr.File(label="Download Final MP4", interactive=False)

                gr.Markdown("### 📊 Editing Analytics")
                with gr.Row():
                    orig_dur_box = gr.Textbox(label="Original Duration", value="--", interactive=False)
                    final_dur_box = gr.Textbox(label="Edited Duration", value="--", interactive=False)
                    scenes_box = gr.Textbox(label="Scenes Detected", value="--", interactive=False)
                    silence_box = gr.Textbox(label="Silences Removed", value="--", interactive=False)

                gr.Markdown("### 📝 Live Processing Log")
                log_box = gr.TextArea(
                    label="System Log",
                    lines=10,
                    max_lines=18,
                    interactive=False,
                    placeholder="Log updates will appear here in real-time...",
                )

        # Wire click event
        process_btn.click(
            fn=process_video,
            inputs=[
                video_input,
                preset_dropdown,
                music_input,
                color_look_dropdown,
                chk_captions,
                chk_color,
                chk_silence,
                chk_face,
                chk_beat,
                format_dropdown,
            ],
            outputs=[
                output_video_player,
                download_file,
                log_box,
                orig_dur_box,
                final_dur_box,
                scenes_box,
                silence_box,
            ],
        )

    return demo


def launch_ui(share: bool = True, server_port: int = 7860):
    """Launch the Gradio Web application."""
    demo = create_gradio_ui()
    demo.queue()
    try:
        demo.launch(share=share, server_port=server_port, inbrowser=True, theme=gr.themes.Soft(), css=CUSTOM_CSS)
    except Exception:
        demo.launch(share=share, server_port=server_port, inbrowser=True)


if __name__ == "__main__":
    launch_ui(share=False)
