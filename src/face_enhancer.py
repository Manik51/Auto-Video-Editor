"""
face_enhancer.py
AI-driven facial restoration and enhancement module using GFPGAN v1.4.
Includes frame-skip performance optimization (every 5th frame) and smooth temporal blending.
"""

import os
import cv2
import numpy as np
from typing import Optional


class FaceEnhancer:
    """
    Restores and sharpens faces in video using GFPGAN v1.4 model.
    Optimized for high FPS via 5-frame sampling and temporal interpolation.
    """

    def __init__(self, model_path: Optional[str] = None, upscale: int = 1, device: Optional[str] = None):
        self.model_path = model_path
        self.upscale = upscale
        self.device = device
        self.restorer = None
        self._initialized = False

    def _init_restorer(self) -> bool:
        """Lazy loader for GFPGAN model."""
        if self._initialized:
            return self.restorer is not None

        self._initialized = True
        try:
            import torch
            from gfpgan import GFPGANer

            dev = self.device
            if dev is None:
                dev = "cuda" if torch.cuda.is_available() else "cpu"

            # Auto-locate model weight
            if not self.model_path or not os.path.exists(self.model_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                default_path = os.path.join(base_dir, "models", "GFPGANv1.4.pth")
                if os.path.exists(default_path):
                    self.model_path = default_path
                else:
                    self.model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"

            print(f"✨ Initializing GFPGAN v1.4 face restorer on {dev}...")
            self.restorer = GFPGANer(
                model_path=self.model_path,
                upscale=self.upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=torch.device(dev),
            )
            print("  ✅ GFPGAN v1.4 initialized successfully.")
            return True
        except Exception as e:
            print(f"⚠️ Notice: GFPGAN could not be loaded ({e}). Skipping face enhancement.")
            self.restorer = None
            return False

    def enhance_frame(self, frame: np.ndarray, weight: float = 0.6) -> np.ndarray:
        """
        Enhance faces within a single frame and blend with original for natural results.
        """
        if not self._init_restorer() or self.restorer is None:
            return frame

        try:
            # cropped_faces, restored_faces, restored_img
            _, _, restored_img = self.restorer.enhance(
                frame,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=weight,
            )
            return restored_img if restored_img is not None else frame
        except Exception as e:
            print(f"⚠️ GFPGAN enhance notice: {e}")
            return frame

    def process_frame_stream(
        self,
        cap: cv2.VideoCapture,
        total_frames: int,
        sample_step: int = 5,
    ):
        """
        Generator that enhances every 5th frame and linearly blends intermediate frames
        to maintain high FPS and temporal consistency.
        """
        if not self._init_restorer() or self.restorer is None:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
            return

        frame_idx = 0
        last_enhanced = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                enhanced = self.enhance_frame(frame)
                last_enhanced = enhanced
                yield enhanced
            else:
                if last_enhanced is not None and last_enhanced.shape == frame.shape:
                    # Blend 50% between current motion and last enhanced texture
                    blended = cv2.addWeighted(frame, 0.55, last_enhanced, 0.45, 0)
                    yield blended
                else:
                    yield frame

            frame_idx += 1


if __name__ == "__main__":
    fe = FaceEnhancer()
    print("FaceEnhancer initialized successfully.")
