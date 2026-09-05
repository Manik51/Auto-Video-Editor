"""
silence_remover.py
Intelligent dead-air and silence remover with frame-accurate cutting,
0.1s speech safety padding, and audio-video synchronization.
"""

import os
import subprocess
import shutil
import tempfile
from typing import List, Tuple, Dict, Any, Optional


class SilenceRemover:
    """
    Removes silent/dead-air segments from video and audio timelines
    while preserving natural speech flow using boundary padding.
    """

    def __init__(self, padding: float = 0.1, min_silence_duration: float = 0.5):
        self.padding = padding
        self.min_silence_duration = min_silence_duration

    def compute_keep_intervals(
        self,
        non_silent_intervals: List[Tuple[float, float]],
        total_duration: float,
    ) -> List[Tuple[float, float]]:
        """
        Apply 0.1s padding before and after each non-silent interval,
        then merge overlapping intervals.
        """
        if not non_silent_intervals:
            return [(0.0, total_duration)]

        padded = []
        for start, end in non_silent_intervals:
            p_start = max(0.0, start - self.padding)
            p_end = min(total_duration, end + self.padding)
            padded.append((p_start, p_end))

        # Merge overlapping or touching intervals
        merged = []
        curr_start, curr_end = padded[0]

        for s, e in padded[1:]:
            if s <= curr_end:  # Overlaps or touches
                curr_end = max(curr_end, e)
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = s, e
        merged.append((curr_start, curr_end))

        # Filter out intervals that are excessively tiny (< 0.2s)
        clean_intervals = [(s, e) for s, e in merged if (e - s) >= 0.2]
        return clean_intervals if clean_intervals else [(0.0, total_duration)]

    def remove_silence(
        self,
        input_video_path: str,
        output_video_path: str,
        keep_intervals: List[Tuple[float, float]],
    ) -> bool:
        """
        Extract and concatenate non-silent video/audio segments using FFmpeg filter_complex
        for frame-accurate, click-free audio-video synchronization.
        """
        if not keep_intervals:
            return False

        # If keeping the entire video (no silence to cut)
        if len(keep_intervals) == 1 and keep_intervals[0][0] == 0.0:
            shutil.copy2(input_video_path, output_video_path)
            return True

        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"

        # Build FFmpeg filter_complex for seamless multi-segment trim and concat
        # Example for N segments:
        # [0:v]trim=start=s1:end=e1,setpts=PTS-STARTPTS[v0];
        # [0:a]atrim=start=s1:end=e1,asetpts=PTS-STARTPTS[a0];
        # ...
        # [v0][a0][v1][a1]...concat=n=N:v=1:a=1[outv][outa]

        filter_parts = []
        concat_inputs = []
        num_segments = len(keep_intervals)

        for i, (start, end) in enumerate(keep_intervals):
            dur = end - start
            filter_parts.append(
                f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]"
            )
            filter_parts.append(
                f"[0:a]atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS[a{i}]"
            )
            concat_inputs.append(f"[v{i}][a{i}]")

        filter_complex_str = (
            "; ".join(filter_parts)
            + f"; {''.join(concat_inputs)}concat=n={num_segments}:v=1:a=1[outv][outa]"
        )

        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", input_video_path,
            "-filter_complex", filter_complex_str,
            "-map", "[outv]",
            "-map", "[outa]",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "19",
            "-c:a", "aac",
            "-b:a", "192k",
            output_video_path,
        ]

        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"⚠️ FFmpeg complex filter notice: {res.stderr.decode('utf-8', errors='ignore')[:300]}")
                # Fallback to segment-file concat demuxer if command line was too long
                return self._fallback_concat_segments(input_video_path, output_video_path, keep_intervals)
            return True
        except Exception as e:
            print(f"⚠️ Error executing silence removal: {e}")
            return self._fallback_concat_segments(input_video_path, output_video_path, keep_intervals)

    def _fallback_concat_segments(
        self,
        input_video_path: str,
        output_video_path: str,
        keep_intervals: List[Tuple[float, float]],
    ) -> bool:
        """Fallback method: trim individual segments into temp files, then concat with demuxer."""
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        temp_dir = tempfile.mkdtemp(prefix="silence_trim_")
        segment_files = []

        try:
            for idx, (s, e) in enumerate(keep_intervals):
                dur = e - s
                seg_path = os.path.join(temp_dir, f"seg_{idx:04d}.mp4")
                cmd = [
                    ffmpeg_bin, "-y",
                    "-ss", f"{s:.3f}",
                    "-i", input_video_path,
                    "-t", f"{dur:.3f}",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac",
                    seg_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if os.path.exists(seg_path) and os.path.getsize(seg_path) > 1000:
                    segment_files.append(seg_path)

            if not segment_files:
                return False

            concat_txt = os.path.join(temp_dir, "concat.txt")
            with open(concat_txt, "w", encoding="utf-8") as f:
                for sp in segment_files:
                    f.write(f"file '{sp}'\n")

            cmd_concat = [
                ffmpeg_bin, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_txt,
                "-c", "copy",
                output_video_path
            ]
            res = subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return res.returncode == 0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    remover = SilenceRemover()
    print("SilenceRemover initialized successfully.")
