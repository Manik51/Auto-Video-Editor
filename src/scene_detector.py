"""
scene_detector.py
Smart auto-cutting module using PySceneDetect and multi-factor quality scoring.
Evaluates motion intensity, audio energy, and face presence to keep top quality scenes.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class SceneDetector:
    """
    Intelligent scene boundary detector and content scorer.
    """

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_duration: float = 1.5,
        keep_ratio: float = 0.70,
        motion_threshold: float = 0.02,
    ):
        self.threshold = threshold
        self.min_scene_duration = min_scene_duration
        self.keep_ratio = keep_ratio
        self.motion_threshold = motion_threshold

        # Initialize face detector for scoring
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def detect_raw_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """
        Detect scene boundaries using PySceneDetect ContentDetector.
        Falls back to frame-difference method if PySceneDetect is unavailable.
        """
        scenes = []
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector

            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=self.threshold, min_scene_len=15))
            scene_manager.detect_scenes(video)
            detected = scene_manager.get_scene_list()

            for scene in detected:
                start_sec = scene[0].get_seconds()
                end_sec = scene[1].get_seconds()
                scenes.append((start_sec, end_sec))

        except Exception as e:
            print(f"⚠️ PySceneDetect notice: {e}. Using OpenCV fallback scene splitter...")
            scenes = self._fallback_detect_scenes(video_path)

        # If no cuts found or only 1 single continuous shot longer than 4.5s, create dynamic montage scenes
        if len(scenes) <= 1:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = total_frames / fps if fps > 0 else 0.0
            cap.release()
            if duration > 4.5:
                scenes = []
                # Paced cuts between 3.0s and 4.5s for dynamic video editing
                step = max(3.0, min(4.5, duration / max(2, int(duration // 3.8))))
                curr = 0.0
                while curr < duration:
                    nxt = min(curr + step, duration)
                    if (duration - nxt) < 2.0:
                        nxt = duration
                    scenes.append((curr, nxt))
                    curr = nxt
                    if curr >= duration:
                        break

        return scenes

    def _fallback_detect_scenes(self, video_path: str) -> List[Tuple[float, float]]:
        """Fallback scene detection using frame difference histogram analysis."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scenes = []
        last_cut_frame = 0
        prev_gray = None

        min_len_frames = int(self.min_scene_duration * fps)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % 3 == 0:  # Sample every 3rd frame for speed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (160, 90))

                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    score = np.mean(diff)
                    if score > self.threshold and (frame_idx - last_cut_frame) >= min_len_frames:
                        scenes.append((last_cut_frame / fps, frame_idx / fps))
                        last_cut_frame = frame_idx

                prev_gray = gray
            frame_idx += 1

        cap.release()
        if total_frames > last_cut_frame:
            scenes.append((last_cut_frame / fps, total_frames / fps))

        return scenes

    def score_and_filter_scenes(
        self,
        video_path: str,
        scenes: List[Tuple[float, float]],
        audio_energy_map: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[float, float]]:
        """
        Filter out scenes shorter than min_scene_duration and static scenes.
        Score remaining scenes based on motion, face presence, and audio energy.
        Keep top keep_ratio (e.g. 70%) of scenes.
        """
        if not scenes:
            return []

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        valid_scenes = []
        for s_start, s_end in scenes:
            dur = s_end - s_start
            if dur >= self.min_scene_duration:
                valid_scenes.append((s_start, s_end))

        if not valid_scenes:
            # If all were shorter, return original scenes to prevent empty video
            cap.release()
            return scenes

        scored_records = []

        for s_start, s_end in valid_scenes:
            dur = s_end - s_start
            # Sample 6 frames within scene
            sample_times = np.linspace(s_start + 0.1 * dur, s_end - 0.1 * dur, 6)
            sampled_frames = []

            for st in sample_times:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(st * fps))
                ret, frame = cap.read()
                if ret and frame is not None:
                    sampled_frames.append(frame)

            # 1. Motion Score
            motion_score = 0.0
            if len(sampled_frames) >= 2:
                diffs = []
                for i in range(len(sampled_frames) - 1):
                    f1 = cv2.resize(cv2.cvtColor(sampled_frames[i], cv2.COLOR_BGR2GRAY), (128, 72))
                    f2 = cv2.resize(cv2.cvtColor(sampled_frames[i + 1], cv2.COLOR_BGR2GRAY), (128, 72))
                    diffs.append(np.mean(cv2.absdiff(f1, f2)))
                motion_score = float(np.mean(diffs)) / 255.0

            # Discard static/dead scenes if motion is near zero
            if motion_score < self.motion_threshold and dur > 3.0:
                # Boring static shot
                continue

            # 2. Face Presence Score
            face_score = 0.0
            if self.face_cascade and sampled_frames:
                face_counts = 0
                for f in sampled_frames:
                    small = cv2.resize(f, (320, 180))
                    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=4)
                    if len(faces) > 0:
                        face_counts += 1
                face_score = face_counts / len(sampled_frames)

            # 3. Audio Energy Score
            audio_score = 0.5
            if audio_energy_map and "segments" in audio_energy_map:
                # Approximate audio energy for this segment
                audio_score = self._calc_audio_energy_in_range(s_start, s_end, audio_energy_map)

            # Weighted overall score
            total_score = 0.45 * motion_score + 0.35 * face_score + 0.20 * audio_score

            scored_records.append({
                "interval": (s_start, s_end),
                "motion": motion_score,
                "face": face_score,
                "audio": audio_score,
                "score": total_score,
            })

        cap.release()

        if not scored_records:
            # Fallback if all were dropped by strict motion filter
            return valid_scenes

        # Keep top keep_ratio of scenes
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        keep_count = max(1, int(len(scored_records) * self.keep_ratio))
        selected = scored_records[:keep_count]

        # Re-sort chronologically by start time
        selected.sort(key=lambda x: x["interval"][0])
        final_timeline = [x["interval"] for x in selected]

        print(f"🎬 Scene Detection: {len(scenes)} detected -> {len(final_timeline)} selected (kept top {int(self.keep_ratio*100)}%)")
        return final_timeline

    def _calc_audio_energy_in_range(self, start: float, end: float, audio_energy_map: Dict[str, Any]) -> float:
        """Helper to get normalized audio energy within [start, end]."""
        try:
            times = audio_energy_map.get("times", [])
            energies = audio_energy_map.get("rms", [])
            if len(times) == 0 or len(energies) == 0:
                return 0.5
            mask = (times >= start) & (times <= end)
            if np.any(mask):
                return float(np.mean(energies[mask]))
            return 0.5
        except Exception:
            return 0.5


if __name__ == "__main__":
    detector = SceneDetector()
    print("SceneDetector initialized successfully.")
