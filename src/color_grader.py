"""
color_grader.py
Cinematic color grading engine for AutoVideoEditor.
Implements:
- 3D LUT (.cube) parsing and vectorized 3D table application
- Auto white balance (Gray World) & auto exposure (CLAHE on LAB)
- Saturation boost (+15%) & Contrast boost (+10%)
- Subtle Vignette effect (radial falloff)
- Calibrated 35mm film grain overlay (strength = 0.03)
"""

import os
import cv2
import numpy as np
from typing import Optional, Tuple


class ColorGrader:
    """
    Applies professional cinematic color grading, LUT transforms,
    dynamic exposure balancing, vignette, and film grain.
    """

    def __init__(self, lut_path: Optional[str] = None):
        self.lut_path = lut_path
        self.lut_table: Optional[np.ndarray] = None
        self.lut_size: int = 17
        self._cached_vignette_mask: Optional[np.ndarray] = None
        self._cached_mask_shape: Optional[Tuple[int, int]] = None

        if lut_path and os.path.exists(lut_path):
            self.load_cube_lut(lut_path)

    def load_cube_lut(self, lut_path: str) -> bool:
        """
        Parse standard Adobe .cube 3D LUT file into 3D NumPy array.
        Standard Cube order: R changes fastest, then G, then B.
        """
        try:
            with open(lut_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            size = None
            data = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("LUT_3D_SIZE"):
                    size = int(line.split()[1])
                    continue
                if line.startswith("TITLE") or line.startswith("DOMAIN_"):
                    continue

                parts = line.split()
                if len(parts) == 3:
                    try:
                        r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
                        data.append([r, g, b])
                    except ValueError:
                        continue

            if size is None:
                size = int(round(len(data) ** (1.0 / 3.0)))

            self.lut_size = size
            # Reshape to (size, size, size, 3) where axes correspond to (B, G, R) or (R, G, B)
            # Cube spec: R fastest, G second, B slowest -> shape (B, G, R, 3)
            arr = np.array(data, dtype=np.float32)
            if len(arr) == size * size * size:
                self.lut_table = arr.reshape((size, size, size, 3))
                self.lut_path = lut_path
                return True
            else:
                print(f"⚠️ Warning: LUT data size mismatch ({len(arr)} vs {size**3})")
                return False
        except Exception as e:
            print(f"⚠️ Failed to parse LUT file {lut_path}: {e}")
            return False

    def auto_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Gray World algorithm for automatic white balance correction.
        """
        b, g, r = cv2.split(frame.astype(np.float32))
        avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
        if avg_b == 0 or avg_g == 0 or avg_r == 0:
            return frame

        avg_gray = (avg_b + avg_g + avg_r) / 3.0
        scale_b = min(2.0, avg_gray / avg_b)
        scale_g = min(2.0, avg_gray / avg_g)
        scale_r = min(2.0, avg_gray / avg_r)

        b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
        g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
        r = np.clip(r * scale_r, 0, 255).astype(np.uint8)
        return cv2.merge([b, g, r])

    def auto_exposure_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) on Luminance
        for balanced exposure without blown highlights.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        # Soft blend CLAHE to maintain natural look (70% original, 30% CLAHE)
        blended_l = cv2.addWeighted(l, 0.70, cl, 0.30, 0)
        merged_lab = cv2.merge([blended_l, a, b])
        return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    def boost_saturation_contrast(self, frame: np.ndarray, sat_boost: float = 0.07, cont_boost: float = 0.04) -> np.ndarray:
        """
        Subtle natural vibrance boost (+7%) and gentle contrast (+4%) that preserves natural skin tones.
        """
        # 1. Saturation boost
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + sat_boost), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2. Gentle Contrast boost around midpoint 128
        alpha = 1.0 + cont_boost
        beta = 128 * (1.0 - alpha)
        adjusted = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
        return adjusted

    def apply_clarity(self, frame: np.ndarray, amount: float = 0.4) -> np.ndarray:
        """
        Studio-grade unsharp mask for razor-sharp 4K / 1080p clarity.
        """
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharp = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def apply_lut_to_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply 3D LUT via fast vectorized table lookup.
        """
        if self.lut_table is None:
            return frame

        b = frame[:, :, 0]
        g = frame[:, :, 1]
        r = frame[:, :, 2]

        scale = (self.lut_size - 1) / 255.0
        b_idx = np.clip(np.round(b * scale).astype(np.int32), 0, self.lut_size - 1)
        g_idx = np.clip(np.round(g * scale).astype(np.int32), 0, self.lut_size - 1)
        r_idx = np.clip(np.round(r * scale).astype(np.int32), 0, self.lut_size - 1)

        out_rgb = self.lut_table[b_idx, g_idx, r_idx]
        out_bgr = out_rgb[:, :, ::-1]
        graded = np.clip(out_bgr * 255.0, 0, 255).astype(np.uint8)
        return graded

    def apply_vignette(self, frame: np.ndarray, strength: float = 0.12) -> np.ndarray:
        """
        Very soft optional cinematic vignette (disabled by default for clean look).
        """
        h, w = frame.shape[:2]
        if self._cached_mask_shape != (h, w) or self._cached_vignette_mask is None:
            Y, X = np.ogrid[:h, :w]
            cx, cy = w / 2.0, h / 2.0
            max_dist = np.sqrt(cx ** 2 + cy ** 2)
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) / max_dist
            falloff = np.clip((dist - 0.5) / 0.5, 0.0, 1.0)
            mask = 1.0 - (strength * (falloff ** 2))
            self._cached_vignette_mask = np.dstack([mask, mask, mask]).astype(np.float32)
            self._cached_mask_shape = (h, w)

        vignetted = (frame.astype(np.float32) * self._cached_vignette_mask)
        return np.clip(vignetted, 0, 255).astype(np.uint8)

    def apply_film_grain(self, frame: np.ndarray, strength: float = 0.015) -> np.ndarray:
        """
        Optional film grain (disabled by default for clean modern footage).
        """
        h, w, c = frame.shape
        if not hasattr(self, "_grain_cache") or self._grain_cache is None or self._grain_cache.shape[:2] != (h, w):
            sigma = strength * 255.0
            self._grain_cache = np.random.normal(0, sigma, (h, w, c)).astype(np.float32)

        grainy = frame.astype(np.float32) + self._grain_cache
        return np.clip(grainy, 0, 255).astype(np.uint8)

    def process_frame(
        self,
        frame: np.ndarray,
        apply_wb: bool = True,
        apply_exposure: bool = True,
        apply_lut: bool = True,
        apply_boost: bool = True,
        apply_clarity_fx: bool = True,
        apply_vignette_fx: bool = False,
        apply_grain_fx: bool = False,
    ) -> np.ndarray:
        """
        Execute clean studio color enhancement and sharpening on an individual frame.
        """
        out = frame
        if apply_wb:
            out = self.auto_white_balance(out)
        if apply_exposure:
            out = self.auto_exposure_clahe(out)
        if apply_boost:
            out = self.boost_saturation_contrast(out, sat_boost=0.07, cont_boost=0.04)
        if apply_lut and self.lut_table is not None:
            out = self.apply_lut_to_frame(out)
        if apply_clarity_fx:
            out = self.apply_clarity(out, amount=0.35)
        if apply_vignette_fx:
            out = self.apply_vignette(out, strength=0.10)
        if apply_grain_fx:
            out = self.apply_film_grain(out, strength=0.01)
        return out


if __name__ == "__main__":
    grader = ColorGrader()
    print("ColorGrader initialized successfully.")
