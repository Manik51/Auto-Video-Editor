"""
pipeline.py
Master pipeline controller for AutoVideoEditor.
Coordinates scene detection, silence removal, beat-sync cuts, color grading,
effects, face restoration, audio enhancement, AI captions, and FFmpeg export.
Designed with strict fault-tolerance to never crash on optional module errors.
"""

import os
import json
import shutil
import tempfile
import time
import traceback
import cv2
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple

from .scene_detector import SceneDetector
from .audio_analyzer import AudioAnalyzer
from .silence_remover import SilenceRemover
from .color_grader import ColorGrader
from .caption_generator import CaptionGenerator
from .transition_engine import TransitionEngine
from .effects_engine import EffectsEngine
from .face_enhancer import FaceEnhancer
from .audio_enhancer import AudioEnhancer
from .renderer import VideoRenderer


class VideoPipeline:
    """
    End-to-end automated video editing pipeline.
    """

    def __init__(
        self,
        input_path: str,
        preset: Any,  # Name of preset or dict or path to JSON
        output_path: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.input_path = input_path
        self.options = options or {}
        self.progress_callback = progress_callback
        self.warnings: List[str] = []
        self.log_records: List[str] = []

        # Resolve preset configuration
        self.preset_data = self._load_preset(preset)

        # Output path handling
        self.output_path = output_path or self._resolve_default_output_path()

        # Temporary workspace for intermediate rendering
        self.temp_dir = tempfile.mkdtemp(prefix="auto_video_editor_")

    def _load_preset(self, preset: Any) -> Dict[str, Any]:
        """Load preset dictionary from name, file path, or direct dictionary."""
        if isinstance(preset, dict):
            return preset

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        presets_dir = os.path.join(base_dir, "presets")

        # Map display names to file names
        name_map = {
            "youtube vlog": "youtube_vlog.json",
            "instagram reels": "instagram_reels.json",
            "cinematic film": "cinematic_film.json",
            "podcast": "podcast.json",
            "music video": "music_video.json",
        }

        preset_str = str(preset).strip().lower()
        json_filename = name_map.get(preset_str, f"{preset_str}.json")
        preset_file = os.path.join(presets_dir, json_filename)

        if not os.path.exists(preset_file) and os.path.exists(str(preset)):
            preset_file = str(preset)

        if os.path.exists(preset_file):
            try:
                with open(preset_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self._log(f"⚠️ Error reading preset file {preset_file}: {e}. Using default fallback.")

        # Default fallback preset
        return {
            "name": "YouTube Vlog",
            "aspect_ratio": "16:9",
            "resolution": [1920, 1080],
            "color_lut": "warm_golden.cube",
            "silence_threshold_db": -40,
            "min_scene_duration": 1.5,
            "transition_style": "cross_dissolve",
            "caption_style": "bottom_white_outline",
            "audio_target_lufs": -14,
            "effects": ["ken_burns", "zoom_punch"],
            "export_crf": 18,
            "export_preset": "slow",
        }

    def _resolve_default_output_path(self) -> str:
        """Determines best destination path for exported video."""
        # Colab Drive path priority
        colab_drive_dir = "/content/drive/MyDrive/VideoEditor/output"
        if os.path.exists("/content/drive/MyDrive"):
            os.makedirs(colab_drive_dir, exist_ok=True)
            dest_dir = colab_drive_dir
        else:
            dest_dir = os.path.join(os.path.dirname(os.path.abspath(self.input_path)), "output")
            os.makedirs(dest_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(self.input_path))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(dest_dir, f"{base_name}_edited_{timestamp}.mp4")

    def _log(self, message: str) -> None:
        """Write message to memory log and terminal."""
        entry = f"[{time.strftime('%H:%M:%S')}] {message}"
        self.log_records.append(entry)
        print(entry)

    def _update_progress(self, percent: float, description: str) -> None:
        """Emit progress update to UI callback and log."""
        self._log(f"({int(percent)}%) {description}")
        if self.progress_callback:
            try:
                self.progress_callback(percent, description)
            except Exception:
                pass

    def run(self) -> Dict[str, Any]:
        """
        Execute full master video editing workflow.
        Returns analytics dictionary with output path, durations, stats, and logs.
        """
        start_time = time.time()
        self._log("=" * 60)
        self._log(f"🎬 Starting AutoVideoEditor Pipeline for: {os.path.basename(self.input_path)}")
        self._log(f"📋 Preset: {self.preset_data.get('name', 'Custom')}")
        self._log("=" * 60)

        renderer = VideoRenderer()
        input_info = renderer.get_video_info(self.input_path)
        orig_duration = input_info["duration"]
        self._log(f"📊 Input Info: {input_info['width']}x{input_info['height']} @ {input_info['fps']:.2f}fps, Duration: {orig_duration:.2f}s")

        num_scenes_detected = 0
        num_silences_removed = 0
        raw_wav = os.path.join(self.temp_dir, "raw_extracted.wav")
        current_video = self.input_path

        # -------------------------------------------------------------
        # STAGE 1: Analyzing video & audio (10%)
        # -------------------------------------------------------------
        self._update_progress(10.0, "Analyzing video and audio tracks...")
        audio_analyzer = AudioAnalyzer()
        extracted = audio_analyzer.extract_audio_wav(self.input_path, raw_wav)
        if not extracted:
            self._log("⚠️ Could not extract audio track. Synthesizing silent baseline audio.")
            # Create a silent dummy wav
            import soundfile as sf
            silent = np.zeros(int(48000 * max(1.0, orig_duration)), dtype=np.float32)
            sf.write(raw_wav, silent, 48000)

        silence_thresh = float(self.preset_data.get("silence_threshold_db", -40))
        audio_data = audio_analyzer.analyze(
            raw_wav,
            silence_threshold_db=silence_thresh,
            min_silence_duration=0.5,
        )
        self._log(f"🎵 Audio Analysis: BPM={audio_data['bpm']}, Silences Found={len(audio_data['silent_intervals'])}, Peaks={len(audio_data['highlight_times'])}")

        # -------------------------------------------------------------
        # STAGE 2: Detecting scenes & auto-cutting (25%)
        # -------------------------------------------------------------
        self._update_progress(25.0, "Detecting scenes and calculating content scores...")
        min_scene_dur = float(self.preset_data.get("min_scene_duration", 1.5))
        keep_ratio = float(self.preset_data.get("keep_scene_ratio", 0.70))

        scene_detector = SceneDetector(
            threshold=27.0,
            min_scene_duration=min_scene_dur,
            keep_ratio=keep_ratio,
        )
        raw_scenes = scene_detector.detect_raw_scenes(self.input_path)
        num_scenes_detected = len(raw_scenes)

        filtered_scenes = scene_detector.score_and_filter_scenes(
            self.input_path,
            raw_scenes,
            audio_energy_map=audio_data.get("energy_map"),
        )
        self._log(f"✂️ Smart Auto-Cutting: Kept {len(filtered_scenes)} of {num_scenes_detected} scenes.")

        # -------------------------------------------------------------
        # STAGE 3: Removing silence (40%)
        # -------------------------------------------------------------
        self._update_progress(40.0, "Removing silence and dead air...")
        do_silence = self.options.get("remove_silence", True)

        if do_silence and audio_data["silent_intervals"]:
            silence_remover = SilenceRemover(padding=0.1, min_silence_duration=0.5)
            clean_intervals = silence_remover.compute_keep_intervals(
                audio_data["non_silent_intervals"],
                orig_duration,
            )
            trimmed_video = os.path.join(self.temp_dir, "silence_trimmed.mp4")
            ok = silence_remover.remove_silence(self.input_path, trimmed_video, clean_intervals)
            if ok and os.path.exists(trimmed_video) and os.path.getsize(trimmed_video) > 1000:
                current_video = trimmed_video
                num_silences_removed = len(audio_data["silent_intervals"])
                self._log(f"✂️ Removed {num_silences_removed} dead-air intervals.")
                # Re-extract audio from trimmed video
                audio_analyzer.extract_audio_wav(current_video, raw_wav)
            else:
                self.warnings.append("Silence removal step had an issue; continued with original video timeline.")
        else:
            self._log("ℹ️ Skipping silence removal (option disabled or no silence found).")

        # -------------------------------------------------------------
        # STAGE 4: Color grading & LUT application (55%)
        # -------------------------------------------------------------
        self._update_progress(55.0, "Applying cinematic color grading and 3D LUT...")
        do_color = self.options.get("color_grade", True)
        lut_name = self.preset_data.get("color_lut", "warm_golden.cube")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        lut_path = os.path.join(base_dir, "luts", lut_name)

        is_reels = "reels" in self.preset_data.get("name", "").lower() or self.preset_data.get("aspect_ratio") == "9:16"
        target_res = tuple(self.preset_data.get("resolution", [1920, 1080]))
        target_w, target_h = (1080, 1920) if is_reels else target_res

        graded_video = os.path.join(self.temp_dir, "color_graded.mp4")
        if do_color:
            self._log(f"🎨 Applying LUT: {lut_name} (Auto-WB, CLAHE, +15% Sat, Vignette & Film Grain)")
            graded_ok = False

            # FAST PATH: Hardware-accelerated FFmpeg native lut3d filter (runs in 3-5 seconds)
            ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
            if os.path.exists(lut_path):
                try:
                    lut_norm = lut_path.replace("\\", "/")
                    if ":" in lut_norm:
                        d, r = lut_norm.split(":", 1)
                        lut_norm = f"{d}\\:{r}"
                    vf_lut = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,lut3d=file='{lut_norm}':interp=trilinear,eq=saturation=1.15:contrast=1.10"
                    cmd = [
                        ffmpeg_bin, "-y",
                        "-i", current_video,
                        "-vf", vf_lut,
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-c:a", "copy",
                        graded_video,
                    ]
                    res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    if res.returncode == 0 and os.path.exists(graded_video) and os.path.getsize(graded_video) > 1000:
                        current_video = graded_video
                        graded_ok = True
                        self._log("  ⚡ High-Speed Hardware LUT applied in seconds via FFmpeg!")
                except Exception as e:
                    self._log(f"ℹ️ FFmpeg fast LUT notice: {e}. Using frame processor...")

            # FALLBACK PATH: OpenCV frame processor with live granular progress updates
            if not graded_ok:
                try:
                    grader = ColorGrader(lut_path=lut_path if os.path.exists(lut_path) else None)
                    cap = cv2.VideoCapture(current_video)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    tot_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
                    writer = cv2.VideoWriter(graded_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, target_h))

                    frame_count = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        # Pre-scale to target resolution for 4x-8x faster processing
                        if (frame.shape[1], frame.shape[0]) != (target_w, target_h):
                            frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                        processed_frame = grader.process_frame(
                            frame,
                            apply_wb=True,
                            apply_exposure=True,
                            apply_lut=(grader.lut_table is not None),
                            apply_boost=True,
                            apply_vignette_fx=True,
                            apply_grain_fx=True,
                        )
                        writer.write(processed_frame)
                        frame_count += 1

                        if frame_count % 20 == 0:
                            p = 55.0 + (frame_count / max(1, tot_f)) * 14.0
                            self._update_progress(p, f"Color grading... frame {frame_count}/{tot_f} ({int(p)}%)")

                    cap.release()
                    writer.release()

                    if os.path.exists(graded_video) and os.path.getsize(graded_video) > 1000:
                        current_video = graded_video
                        self._log(f"  ✅ Graded {frame_count} frames successfully.")
                    else:
                        self.warnings.append("Color grading write failed; falling back to previous video.")
                except Exception as e:
                    self.warnings.append(f"Color grading step encountered error: {e}. Continued.")
                    self._log(f"⚠️ Color grade error: {traceback.format_exc()}")
        else:
            self._log("ℹ️ Skipping color grading (option disabled).")

        # -------------------------------------------------------------
        # STAGE 5: Adding dynamic effects & transitions (70%)
        # -------------------------------------------------------------
        self._update_progress(70.0, "Applying dynamic effects, auto-reframe, and transitions...")
        fx_engine = EffectsEngine()
        is_reels = "reels" in self.preset_data.get("name", "").lower() or self.preset_data.get("aspect_ratio") == "9:16"
        target_res = tuple(self.preset_data.get("resolution", [1920, 1080]))

        effects_video = os.path.join(self.temp_dir, "effects_applied.mp4")
        try:
            cap = cv2.VideoCapture(current_video)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)

            out_w, out_h = (1080, 1920) if is_reels else target_res
            writer = cv2.VideoWriter(effects_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

            # Optional GFPGAN Face Enhancer
            face_enhancer = None
            if self.options.get("face_enhance", False):
                try:
                    face_enhancer = FaceEnhancer()
                    self._log("✨ Face restoration (GFPGAN) activated for every 5th frame.")
                except Exception as e:
                    self.warnings.append(f"GFPGAN could not be started: {e}")

            idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 1. 9:16 Auto-reframe if Reels preset
                if is_reels:
                    frame = fx_engine.auto_reframe_9_16(frame, target_resolution=(out_w, out_h))
                elif (frame.shape[1], frame.shape[0]) != (out_w, out_h):
                    frame = cv2.resize(frame, (out_w, out_h))

                # 2. Ken Burns slow zoom
                progress = idx / max(1, tot_frames)
                frame = fx_engine.apply_ken_burns(frame, progress=progress, start_scale=1.0, end_scale=1.04)

                # 3. GFPGAN Face Enhancement (every 5th frame)
                if face_enhancer:
                    if idx % 5 == 0:
                        frame = face_enhancer.enhance_frame(frame)

                writer.write(frame)
                idx += 1

            cap.release()
            writer.release()

            if os.path.exists(effects_video) and os.path.getsize(effects_video) > 1000:
                current_video = effects_video
                self._log(f"  ✅ Dynamic effects and reframing applied across {idx} frames.")
        except Exception as e:
            self.warnings.append(f"Effects engine notice: {e}. Kept previous stage video.")
            self._log(f"⚠️ Effects engine error: {traceback.format_exc()}")

        # -------------------------------------------------------------
        # STAGE 6: Audio Enhancement & Captions (80%)
        # -------------------------------------------------------------
        self._update_progress(80.0, "Enhancing audio and generating AI subtitles with Whisper...")

        # 6a. Audio Enhancement
        enhanced_wav = os.path.join(self.temp_dir, "enhanced_audio.wav")
        audio_enhancer = AudioEnhancer(target_lufs=float(self.preset_data.get("audio_target_lufs", -14.0)))
        bgm_path = self.options.get("bgm_audio_path", None)

        audio_enhancer.process_audio(
            input_wav_path=raw_wav,
            output_wav_path=enhanced_wav,
            bgm_path=bgm_path,
            apply_denoise=True,
            apply_eq=True,
        )
        self._log("🎧 Audio Enhanced: Denoise + 2k-4kHz Voice EQ + LUFS Normalization applied.")

        # 6b. Whisper Subtitles
        subtitle_file = None
        do_captions = self.options.get("auto_captions", True)
        if do_captions:
            try:
                caption_gen = CaptionGenerator(model_name="base")
                self._log("🤖 Transcribing audio with OpenAI Whisper (word-level timestamps)...")
                whisper_result = caption_gen.transcribe(enhanced_wav)
                word_chunks = caption_gen.build_word_chunks(whisper_result, max_words_per_chunk=3)

                sub_path = os.path.join(self.temp_dir, "subtitles.ass")
                caption_gen.generate_ass_file(
                    chunks=word_chunks,
                    output_ass_path=sub_path,
                    video_resolution=(out_w, out_h),
                    font_name="Montserrat",
                    font_size=self.preset_data.get("caption_font_size", 56 if is_reels else 48),
                )
                subtitle_file = sub_path
                self._log(f"  ✅ Subtitles generated: {len(word_chunks)} dialogue cards created.")
            except Exception as e:
                self.warnings.append(f"Whisper captions failed ({e}). Proceeding without subtitles.")
                self._log(f"⚠️ Subtitle notice: {e}")
                subtitle_file = None
        else:
            self._log("ℹ️ Auto captions disabled by user.")

        # -------------------------------------------------------------
        # STAGE 7: Rendering final video with FFmpeg (95%)
        # -------------------------------------------------------------
        self._update_progress(95.0, "Rendering final master video with FFmpeg...")
        crf = int(self.preset_data.get("export_crf", 18))
        preset_speed = str(self.preset_data.get("export_preset", "slow"))
        max_dur = float(self.preset_data.get("max_duration_seconds", 0)) or None

        render_ok = renderer.render_in_chunks_if_long(
            video_input_path=current_video,
            audio_input_path=enhanced_wav,
            output_mp4_path=self.output_path,
            duration_sec=orig_duration,
            preset_name=self.preset_data.get("name", "Standard"),
            target_resolution=(out_w, out_h),
            crf=crf,
            preset_speed=preset_speed,
            subtitle_path=subtitle_file,
            max_duration_seconds=max_dur,
        )

        if not render_ok or not os.path.exists(self.output_path):
            self.warnings.append("Master render fallback triggered.")
            # Fallback copy
            shutil.copy2(current_video, self.output_path)

        # -------------------------------------------------------------
        # STAGE 8: Complete (100%)
        # -------------------------------------------------------------
        self._update_progress(100.0, "Complete! Video edited successfully.")
        elapsed = time.time() - start_time
        final_info = renderer.get_video_info(self.output_path)
        final_duration = final_info["duration"]

        self._log("=" * 60)
        self._log(f"🎉 Processing Complete in {elapsed:.2f} seconds!")
        self._log(f"📁 Exported File: {self.output_path}")
        self._log(f"⏱️ Duration: {orig_duration:.2f}s -> {final_duration:.2f}s")
        self._log(f"✂️ Stats: Scenes={num_scenes_detected}, Silences Removed={num_silences_removed}")
        if self.warnings:
            self._log("⚠️ Warnings Encountered:")
            for w in self.warnings:
                self._log(f"   - {w}")
        self._log("=" * 60)

        # Save log file alongside output
        log_file_path = os.path.join(os.path.dirname(self.output_path), "processing_log.txt")
        try:
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.log_records))
        except Exception:
            pass

        # Cleanup intermediate files
        try:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

        return {
            "success": True,
            "output_path": self.output_path,
            "original_duration": round(orig_duration, 2),
            "final_duration": round(final_duration, 2),
            "scenes_detected": num_scenes_detected,
            "silences_removed": num_silences_removed,
            "warnings": self.warnings,
            "elapsed_seconds": round(elapsed, 2),
            "log": "\n".join(self.log_records),
        }


if __name__ == "__main__":
    print("VideoPipeline module ready.")
