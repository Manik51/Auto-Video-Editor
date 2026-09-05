"""
caption_generator.py
Whisper AI auto-subtitling module with word-level timing,
CapCut-style dynamic subtitle formatting, and FFmpeg burning.
"""

import os
import re
import shutil
import subprocess
from typing import Dict, List, Any, Optional, Tuple


class CaptionGenerator:
    """
    Transcribes audio with OpenAI Whisper, generates styled ASS/SRT subtitles,
    and burns them into video using FFmpeg.
    """

    def __init__(self, model_name: str = "base", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _get_model(self):
        """Lazy loader for Whisper model."""
        if self._model is None:
            import whisper
            import torch

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"🎙️ Loading Whisper AI ({self.model_name}) on {device}...")
            self._model = whisper.load_model(self.model_name, device=device)
        return self._model

    def transcribe(self, audio_wav_path: str) -> Dict[str, Any]:
        """
        Transcribe audio and extract word-level timestamps.
        """
        model = self._get_model()
        result = model.transcribe(
            audio_wav_path,
            word_timestamps=True,
            verbose=False,
            fp16=(self.device != "cpu"),
        )
        return result

    def format_timestamp_ass(self, seconds: float) -> str:
        """Format seconds into ASS timestamp format: H:MM:SS.cc"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centisecs = int(round((seconds - int(seconds)) * 100))
        if centisecs >= 100:
            secs += 1
            centisecs = 0
        return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

    def format_timestamp_srt(self, seconds: float) -> str:
        """Format seconds into SRT timestamp format: HH:MM:SS,mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        if millis >= 1000:
            secs += 1
            millis = 0
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def build_word_chunks(
        self,
        whisper_result: Dict[str, Any],
        max_words_per_chunk: int = 4,
        max_duration_sec: float = 2.2,
    ) -> List[Dict[str, Any]]:
        """
        Group words into punchy CapCut-style 2-4 word phrases for high viewer retention.
        """
        chunks = []
        segments = whisper_result.get("segments", [])

        for seg in segments:
            words = seg.get("words", [])
            if not words:
                # Fallback to segment-level text if word timestamps are absent
                chunks.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                })
                continue

            curr_chunk_words = []
            chunk_start = None

            for w in words:
                word_text = w.get("word", "").strip()
                if not word_text:
                    continue

                w_start = w.get("start", seg["start"])
                w_end = w.get("end", seg["end"])

                if chunk_start is None:
                    chunk_start = w_start

                curr_chunk_words.append(word_text)

                # Condition to flush chunk: hit word limit, punctuation break, or duration threshold
                has_punct = word_text.endswith((".", "!", "?", ",", "—", "-"))
                dur = w_end - chunk_start
                if len(curr_chunk_words) >= max_words_per_chunk or has_punct or dur >= max_duration_sec:
                    chunks.append({
                        "start": chunk_start,
                        "end": w_end,
                        "text": " ".join(curr_chunk_words),
                    })
                    curr_chunk_words = []
                    chunk_start = None

            if curr_chunk_words and chunk_start is not None:
                chunks.append({
                    "start": chunk_start,
                    "end": words[-1].get("end", seg["end"]),
                    "text": " ".join(curr_chunk_words),
                })

        return chunks

    def generate_ass_file(
        self,
        chunks: List[Dict[str, Any]],
        output_ass_path: str,
        video_resolution: Tuple[int, int] = (1920, 1080),
        font_name: str = "Montserrat",
        font_size: Optional[int] = None,
    ) -> str:
        """
        Generate an Advanced SubStation Alpha (.ass) subtitle file with professional CapCut styling.
        White text, solid black outline, 10% bottom margin, bold font.
        """
        width, height = video_resolution
        is_portrait = height > width

        if font_size is None:
            font_size = 56 if is_portrait else 48

        margin_v = int(height * 0.10)  # 10% from bottom
        outline_width = 3.5 if not is_portrait else 4.0

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: CapCutStyle,{font_name},{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},1.2,2,30,30,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        for c in chunks:
            start_t = self.format_timestamp_ass(c["start"])
            end_t = self.format_timestamp_ass(c["end"])
            # Escape curly braces for ASS
            clean_text = c["text"].replace("{", "\\{").replace("}", "\\}")
            # Uppercase for bold modern social media aesthetic
            display_text = clean_text.upper()
            ass_content += f"Dialogue: 0,{start_t},{end_t},CapCutStyle,,0,0,0,,{display_text}\n"

        with open(output_ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        return output_ass_path

    def generate_srt_file(self, chunks: List[Dict[str, Any]], output_srt_path: str) -> str:
        """
        Generate standard .srt subtitle file.
        """
        lines = []
        for idx, c in enumerate(chunks, start=1):
            s_t = self.format_timestamp_srt(c["start"])
            e_t = self.format_timestamp_srt(c["end"])
            lines.append(f"{idx}\n{s_t} --> {e_t}\n{c['text'].upper()}\n")

        with open(output_srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return output_srt_path

    def burn_subtitles(
        self,
        input_video_path: str,
        subtitle_path: str,
        output_video_path: str,
        fonts_dir: Optional[str] = None,
    ) -> bool:
        """
        Hardcode/burn subtitles into video frames using FFmpeg subtitles filter.
        """
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        sub_norm = subtitle_path.replace("\\", "/")
        # On Windows, escape colon after drive letter: C:/ -> C\:/
        if ":" in sub_norm:
            drive, rest = sub_norm.split(":", 1)
            sub_norm = f"{drive}\\:{rest}"

        vf_filter = f"subtitles='{sub_norm}'"
        if fonts_dir and os.path.exists(fonts_dir):
            f_norm = fonts_dir.replace("\\", "/")
            if ":" in f_norm:
                fdrive, frest = f_norm.split(":", 1)
                f_norm = f"{fdrive}\\:{frest}"
            vf_filter += f":fontsdir='{f_norm}'"

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", input_video_path,
            "-vf", vf_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "copy",
            output_video_path,
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(output_video_path):
                return True
            else:
                print(f"⚠️ Subtitle burn notice: {res.stderr.decode('utf-8', errors='ignore')[:300]}")
                return False
        except Exception as e:
            print(f"⚠️ Error burning subtitles: {e}")
            return False


if __name__ == "__main__":
    cg = CaptionGenerator()
    print("CaptionGenerator initialized successfully.")
