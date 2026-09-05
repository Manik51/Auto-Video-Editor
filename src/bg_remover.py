"""
bg_remover.py
AI background removal, replacement, and bokeh depth blur using rembg.
"""

import cv2
import numpy as np
from typing import Optional


class BackgroundRemover:
    """
    Removes or blurs backgrounds using rembg (U2-Net) for talking head isolation.
    """

    def __init__(self, model_name: str = "u2net"):
        self.model_name = model_name
        self.session = None
        self._initialized = False

    def _init_session(self) -> bool:
        if self._initialized:
            return self.session is not None
        self._initialized = True
        try:
            from rembg import new_session
            print(f"🪄 Initializing background remover ({self.model_name})...")
            self.session = new_session(self.model_name)
            return True
        except Exception as e:
            print(f"⚠️ Notice: rembg could not be initialized: {e}")
            self.session = None
            return False

    def remove_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Removes background returning BGRA frame with transparent background.
        """
        if not self._init_session() or self.session is None:
            # Fallback: return 4-channel BGRA with full alpha
            b, g, r = cv2.split(frame)
            a = np.ones_like(b) * 255
            return cv2.merge([b, g, r, a])

        try:
            from rembg import remove
            from PIL import Image

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            out_pil = remove(pil_img, session=self.session)
            out_arr = np.array(out_pil)
            # RGBA to BGRA
            bgra = cv2.cvtColor(out_arr, cv2.COLOR_RGBA2BGRA)
            return bgra
        except Exception as e:
            print(f"⚠️ rembg execution error: {e}")
            b, g, r = cv2.split(frame)
            a = np.ones_like(b) * 255
            return cv2.merge([b, g, r, a])

    def blur_background(self, frame: np.ndarray, blur_strength: int = 35) -> np.ndarray:
        """
        Applies a cinematic bokeh blur to the background while keeping the subject sharp.
        """
        if blur_strength % 2 == 0:
            blur_strength += 1

        bgra = self.remove_background(frame)
        alpha = bgra[:, :, 3].astype(np.float32) / 255.0
        alpha_3d = np.dstack([alpha, alpha, alpha])

        # Create heavily blurred background
        blurred_bg = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)

        # Composite foreground over blurred background
        composite = (frame.astype(np.float32) * alpha_3d) + (blurred_bg.astype(np.float32) * (1.0 - alpha_3d))
        return np.clip(composite, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    remover = BackgroundRemover()
    print("BackgroundRemover initialized successfully.")
