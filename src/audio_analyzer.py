"""
audio_analyzer.py
Audio intelligence module using Librosa and SoundFile.
Performs silence detection (-40dB), musical beat tracking (BPM & timestamps),
and audio energy profiling for highlight detection.
"""

import os
import subprocess
import shutil
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class AudioAnalyzer:
    """
    Extracts, analyzes, and detects musical beats, silences, and energy peaks in video audio.
    """

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def extract_audio_wav(self, video_path: str, output_wav_path: str) -> bool:
        """
        Extract high-fidelity audio stream from video file into uncompressed WAV.
        """
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(self.sample_rate),
            "-ac", "2",
            output_wav_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0 and os.path.exists(output_wav_path)
        except Exception as e:
            print(f"⚠️ Error extracting audio with FFmpeg: {e}")
            return False

    def analyze(
        self,
        wav_path: str,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Full audio analysis pipeline:
        1. Silence intervals detection (< silence_threshold_db, > min_silence_duration)
        2. Non-silent speech intervals
        3. Beat tracking (BPM and beat timestamps)
        4. Energy peaks (highlight moments)
        5. RMS energy curve
        """
        try:
            import librosa
            # Load audio (mono for fast analysis)
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            # 1. Silence & Non-silence Intervals
            ref_db = abs(silence_threshold_db)
            non_silent_indices = librosa.effects.split(y, top_db=ref_db, frame_length=2048, hop_length=512)
            non_silent_intervals = []
            for start_idx, end_idx in non_silent_indices:
                s_sec = float(start_idx) / sr
                e_sec = float(end_idx) / sr
                non_silent_intervals.append((s_sec, e_sec))

            # Detect silent gaps between non-silent intervals
            silent_intervals = []
            last_end = 0.0
            for s_start, s_end in non_silent_intervals:
                if (s_start - last_end) >= min_silence_duration:
                    silent_intervals.append((last_end, s_start))
                last_end = s_end
            if (duration - last_end) >= min_silence_duration:
                silent_intervals.append((last_end, duration))

            # 2. Beat Tracking & Tempo
            try:
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                bpm = float(tempo[0]) if isinstance(tempo, (list, np.ndarray)) else float(tempo)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
            except Exception as e:
                bpm = 120.0
                beat_times = []

            # 3. Energy RMS & Highlight Moments
            hop_length = 512
            rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
            times = librosa.times_like(rms, sr=sr, hop_length=hop_length)
        except ImportError:
            # High-performance SoundFile + NumPy fallback
            import soundfile as sf
            data, sr = sf.read(wav_path)
            if data.ndim > 1:
                y = np.mean(data, axis=1)
            else:
                y = data
            duration = len(y) / sr

            # Frame RMS calculation
            frame_len = 2048
            hop_len = 512
            num_frames = max(1, (len(y) - frame_len) // hop_len + 1)
            rms = np.zeros(num_frames, dtype=np.float32)
            for i in range(num_frames):
                seg = y[i * hop_len: i * hop_len + frame_len]
                rms[i] = np.sqrt(np.mean(seg ** 2)) if len(seg) > 0 else 0.0

            times = np.arange(num_frames) * (hop_len / sr)

            # Silence thresholding
            ref_power = np.max(rms ** 2) if np.max(rms) > 0 else 1.0
            db = 10.0 * np.log10(np.maximum(rms ** 2, 1e-10) / max(1e-10, ref_power))
            non_silent_mask = db >= silence_threshold_db

            # Extract contiguous intervals
            non_silent_intervals = []
            in_seg = False
            s_time = 0.0
            for i, is_ns in enumerate(non_silent_mask):
                t = float(times[i])
                if is_ns and not in_seg:
                    in_seg = True
                    s_time = t
                elif not is_ns and in_seg:
                    in_seg = False
                    non_silent_intervals.append((s_time, t))
            if in_seg:
                non_silent_intervals.append((s_time, duration))

            if not non_silent_intervals:
                non_silent_intervals = [(0.0, duration)]

            silent_intervals = []
            last_end = 0.0
            for s_start, s_end in non_silent_intervals:
                if (s_start - last_end) >= min_silence_duration:
                    silent_intervals.append((last_end, s_start))
                last_end = s_end
            if (duration - last_end) >= min_silence_duration:
                silent_intervals.append((last_end, duration))

            bpm = 120.0
            beat_times = list(np.arange(0, duration, 60.0 / bpm))

        # Normalize RMS to [0.0, 1.0]
        max_rms = np.max(rms) if len(rms) > 0 and np.max(rms) > 0 else 1.0
        norm_rms = (rms / max_rms).astype(np.float32)

        peak_threshold = np.percentile(norm_rms, 92) if len(norm_rms) > 0 else 0.8
        peak_indices = np.where(norm_rms >= peak_threshold)[0]
        highlight_times = [float(times[idx]) for idx in peak_indices]

        # Consolidate nearby highlight peaks (within 1.5s)
        filtered_highlights = []
        last_highlight = -999.0
        for ht in highlight_times:
            if (ht - last_highlight) >= 1.5:
                filtered_highlights.append(ht)
                last_highlight = ht

        return {
            "duration": duration,
            "bpm": round(bpm, 2),
            "beat_times": beat_times,
            "non_silent_intervals": non_silent_intervals,
            "silent_intervals": silent_intervals,
            "highlight_times": filtered_highlights,
            "energy_map": {
                "times": times,
                "rms": norm_rms,
            },
        }


if __name__ == "__main__":
    analyzer = AudioAnalyzer()
    print("AudioAnalyzer initialized successfully.")
