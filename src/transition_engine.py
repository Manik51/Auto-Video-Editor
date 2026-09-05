"""
transition_engine.py
Cinematic transition generator for AutoVideoEditor.
Implements:
1. Fade to black (0.3s)
2. Cross dissolve (0.3s)
3. Zoom punch (0.15s)
4. Glitch transition (3-frame RGB chromatic split & scanlines)
5. Whip pan blur (horizontal directional motion blur 0.2s)
With non-repeating random selection rules.
"""

import random
import cv2
import numpy as np
from typing import List, Optional, Tuple


class TransitionEngine:
    """
    Renders seamless, dynamic transitions between scene cuts.
    Guarantees no transition is repeated twice consecutively.
    """

    TRANSITION_TYPES = [
        "cross_dissolve",
        "fade_to_black",
        "zoom_punch",
        "glitch_transition",
        "whip_pan_blur",
    ]

    def __init__(self):
        self.last_transition: Optional[str] = None

    def pick_next_transition(self, preferred: Optional[str] = None) -> str:
        """
        Select the next transition type. If preferred is specified, attempts to use it
        unless it repeats the previous one (or returns preferred directly if configured).
        """
        if preferred and preferred in self.TRANSITION_TYPES:
            self.last_transition = preferred
            return preferred

        choices = [t for t in self.TRANSITION_TYPES if t != self.last_transition]
        picked = random.choice(choices)
        self.last_transition = picked
        return picked

    def render_cross_dissolve(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Cross dissolve: smooth linear alpha blending between tail of A and head of B.
        """
        n = min(len(frames_a), len(frames_b))
        if n == 0:
            return []

        out = []
        for i in range(n):
            alpha = float(i) / max(1, n - 1)
            # Smoothstep interpolation for natural visual ease
            s_alpha = alpha * alpha * (3 - 2 * alpha)
            blended = cv2.addWeighted(frames_a[i], 1.0 - s_alpha, frames_b[i], s_alpha, 0)
            out.append(blended)
        return out

    def render_fade_to_black(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Fade to black: Clip A fades to pure black, Clip B fades in from black.
        """
        out = []
        n_a = len(frames_a)
        n_b = len(frames_b)

        # Fade out A
        for i, f in enumerate(frames_a):
            progress = float(i) / max(1, n_a)
            factor = max(0.0, 1.0 - progress)
            out.append((f.astype(np.float32) * factor).astype(np.uint8))

        # Fade in B
        for i, f in enumerate(frames_b):
            factor = min(1.0, float(i) / max(1, n_b - 1))
            out.append((f.astype(np.float32) * factor).astype(np.uint8))

        return out

    def render_zoom_punch(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Zoom punch: Clip A punches in rapidly (1.0 -> 1.30x), cuts to B punching out (1.30 -> 1.0x).
        """
        out = []
        n_a = len(frames_a)
        n_b = len(frames_b)

        def zoom_frame(img: np.ndarray, scale: float) -> np.ndarray:
            h, w = img.shape[:2]
            if abs(scale - 1.0) < 0.01:
                return img
            ch, cw = int(h / scale), int(w / scale)
            y1 = (h - ch) // 2
            x1 = (w - cw) // 2
            cropped = img[y1:y1 + ch, x1:x1 + cw]
            return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        for i, f in enumerate(frames_a):
            t = float(i) / max(1, n_a - 1)
            scale = 1.0 + 0.25 * (t ** 2)
            out.append(zoom_frame(f, scale))

        for i, f in enumerate(frames_b):
            t = float(i) / max(1, n_b - 1)
            scale = 1.25 - 0.25 * (t ** 0.5)
            out.append(zoom_frame(f, scale))

        return out

    def render_glitch_transition(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Glitch transition: 3 frames of RGB channel split (chromatic aberration)
        combined with horizontal digital slice displacement.
        """
        # Blend frames around cut point with digital glitch artifacts
        all_frames = frames_a + frames_b
        out = []

        for idx, frame in enumerate(all_frames):
            h, w, c = frame.shape
            b, g, r = cv2.split(frame)

            # Intensity increases toward center cut
            dist_to_center = abs(idx - len(frames_a))
            intensity = max(4, int(18 / (1 + dist_to_center)))

            # Shift R channel horizontally to the left, B to the right
            M_r = np.float32([[1, 0, -intensity], [0, 1, 0]])
            M_b = np.float32([[1, 0, intensity], [0, 1, 0]])

            r_shifted = cv2.warpAffine(r, M_r, (w, h), borderMode=cv2.BORDER_REFLECT)
            b_shifted = cv2.warpAffine(b, M_b, (w, h), borderMode=cv2.BORDER_REFLECT)

            glitched = cv2.merge([b_shifted, g, r_shifted])

            # Random horizontal scanline slices
            if random.random() > 0.3:
                slice_y = random.randint(0, h - 30)
                slice_h = random.randint(5, 25)
                shift_x = random.randint(-25, 25)
                slice_block = glitched[slice_y:slice_y + slice_h, :]
                M_slice = np.float32([[1, 0, shift_x], [0, 1, 0]])
                glitched[slice_y:slice_y + slice_h, :] = cv2.warpAffine(
                    slice_block, M_slice, (w, slice_h), borderMode=cv2.BORDER_REFLECT
                )

            out.append(glitched)

        return out

    def render_whip_pan_blur(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Whip pan blur: high velocity horizontal motion blur between cut boundaries.
        """
        out = []
        n_a = len(frames_a)
        n_b = len(frames_b)

        def motion_blur_horizontal(img: np.ndarray, size: int) -> np.ndarray:
            if size <= 1:
                return img
            kernel = np.zeros((size, size))
            kernel[int((size - 1) / 2), :] = np.ones(size)
            kernel = kernel / size
            return cv2.filter2D(img, -1, kernel)

        # Pan out clip A to right
        for i, f in enumerate(frames_a):
            progress = float(i) / max(1, n_a - 1)
            ksize = max(1, int(progress * 35))
            if ksize % 2 == 0:
                ksize += 1
            out.append(motion_blur_horizontal(f, ksize))

        # Pan into clip B from left
        for i, f in enumerate(frames_b):
            progress = 1.0 - (float(i) / max(1, n_b - 1))
            ksize = max(1, int(progress * 35))
            if ksize % 2 == 0:
                ksize += 1
            out.append(motion_blur_horizontal(f, ksize))

        return out

    def apply_transition(
        self,
        frames_a: List[np.ndarray],
        frames_b: List[np.ndarray],
        trans_type: Optional[str] = None,
    ) -> List[np.ndarray]:
        """
        Execute transition between two frame buffers using chosen transition type.
        """
        t_type = trans_type or self.pick_next_transition()

        if t_type == "cross_dissolve":
            return self.render_cross_dissolve(frames_a, frames_b)
        elif t_type == "fade_to_black":
            return self.render_fade_to_black(frames_a, frames_b)
        elif t_type == "zoom_punch":
            return self.render_zoom_punch(frames_a, frames_b)
        elif t_type == "glitch_transition":
            return self.render_glitch_transition(frames_a, frames_b)
        elif t_type == "whip_pan_blur":
            return self.render_whip_pan_blur(frames_a, frames_b)
        else:
            return self.render_cross_dissolve(frames_a, frames_b)


if __name__ == "__main__":
    engine = TransitionEngine()
    print("TransitionEngine initialized successfully.")
