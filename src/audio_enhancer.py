"""
audio_enhancer.py
Professional audio post-processing engine for AutoVideoEditor.
Implements:
1. Spectral background noise reduction via noisereduce
2. Speech clarity EQ boost in the 2kHz - 4kHz presence range
3. EBU R128 / YouTube standard -14 LUFS loudness normalization
4. Dynamic background music ducking (-18dB) with fade-in and fade-out
"""

import os
import subprocess
import shutil
import numpy as np
import scipy.signal
import soundfile as sf
from typing import Optional


class AudioEnhancer:
    """
    Studio-grade audio post-processing pipeline for crystal clear voiceovers.
    """

    def __init__(self, target_lufs: float = -14.0):
        self.target_lufs = target_lufs

    def reduce_noise(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Removes background noise, HVAC hum, and fan hiss using spectral gating.
        """
        try:
            import noisereduce as nr
            # If stereo (shape [samples, 2] or [2, samples]), process each channel
            if audio_data.ndim == 2:
                if audio_data.shape[0] == 2:  # [2, samples]
                    ch1 = nr.reduce_noise(y=audio_data[0], sr=sample_rate, stationary=True, prop_decrease=0.8)
                    ch2 = nr.reduce_noise(y=audio_data[1], sr=sample_rate, stationary=True, prop_decrease=0.8)
                    return np.vstack([ch1, ch2])
                else:  # [samples, 2]
                    ch1 = nr.reduce_noise(y=audio_data[:, 0], sr=sample_rate, stationary=True, prop_decrease=0.8)
                    ch2 = nr.reduce_noise(y=audio_data[:, 1], sr=sample_rate, stationary=True, prop_decrease=0.8)
                    return np.column_stack([ch1, ch2])
            else:
                return nr.reduce_noise(y=audio_data, sr=sample_rate, stationary=True, prop_decrease=0.8)
        except Exception as e:
            print(f"⚠️ Noise reduction notice: {e}. Keeping raw audio.")
            return audio_data

    def boost_voice_eq(self, audio_data: np.ndarray, sample_rate: int, boost_db: float = 3.5) -> np.ndarray:
        """
        Boosts speech presence frequencies in the 2.0 kHz to 4.0 kHz band
        to give voices professional broadcast clarity.
        """
        try:
            nyquist = sample_rate / 2.0
            center_freq = 3000.0 / nyquist
            bandwidth = 1500.0 / nyquist
            q = center_freq / bandwidth

            # Design digital peaking/notch IIR filter
            b, a = scipy.signal.iirpeak(center_freq, q)

            gain = 10.0 ** (boost_db / 20.0) - 1.0

            if audio_data.ndim == 2:
                if audio_data.shape[0] == 2:
                    voice_band0 = scipy.signal.lfilter(b, a, audio_data[0])
                    voice_band1 = scipy.signal.lfilter(b, a, audio_data[1])
                    boosted = audio_data + gain * np.vstack([voice_band0, voice_band1])
                else:
                    voice_band0 = scipy.signal.lfilter(b, a, audio_data[:, 0])
                    voice_band1 = scipy.signal.lfilter(b, a, audio_data[:, 1])
                    boosted = audio_data + gain * np.column_stack([voice_band0, voice_band1])
            else:
                voice_band = scipy.signal.lfilter(b, a, audio_data)
                boosted = audio_data + gain * voice_band

            # Peak limiter to avoid digital clipping
            max_val = np.max(np.abs(boosted))
            if max_val > 0.98:
                boosted = boosted * (0.98 / max_val)

            return boosted
        except Exception as e:
            print(f"⚠️ Voice EQ notice: {e}")
            return audio_data

    def normalize_lufs_ffmpeg(self, input_wav_path: str, output_wav_path: str) -> bool:
        """
        Normalize loudness to target LUFS (default -14 LUFS) using FFmpeg loudnorm filter.
        """
        ffmpeg_bin = shutil.which("ffmpeg") or "ffmpeg"
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i", input_wav_path,
            "-af", f"loudnorm=I={self.target_lufs:.1f}:TP=-1.5:LRA=11",
            "-ar", "48000",
            output_wav_path,
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return res.returncode == 0 and os.path.exists(output_wav_path)
        except Exception as e:
            print(f"⚠️ Loudness normalization error: {e}")
            shutil.copy2(input_wav_path, output_wav_path)
            return True

    def mix_background_music(
        self,
        voice_wav_path: str,
        bgm_path: str,
        output_wav_path: str,
        bgm_volume_db: float = -18.0,
        fade_in_sec: float = 1.5,
        fade_out_sec: float = 2.0,
    ) -> bool:
        """
        Mixes background music underneath voice track:
        - Ducks music to -18dB relative to voice
        - Applies fade-in and fade-out
        - Automatically matches voice track length
        """
        try:
            v_data, sr = sf.read(voice_wav_path)
            bgm_data, bgm_sr = sf.read(bgm_path)

            # Resample BGM if sample rates differ
            if bgm_sr != sr:
                import librosa
                if bgm_data.ndim == 2:
                    bgm_data = librosa.resample(bgm_data.T, orig_sr=bgm_sr, target_sr=sr).T
                else:
                    bgm_data = librosa.resample(bgm_data, orig_sr=bgm_sr, target_sr=sr)

            # Convert both to stereo
            if v_data.ndim == 1:
                v_data = np.column_stack([v_data, v_data])
            if bgm_data.ndim == 1:
                bgm_data = np.column_stack([bgm_data, bgm_data])

            v_len = len(v_data)

            # Loop or trim BGM to match voice length exactly
            if len(bgm_data) < v_len:
                repeats = int(np.ceil(v_len / len(bgm_data)))
                bgm_data = np.tile(bgm_data, (repeats, 1))
            bgm_data = bgm_data[:v_len]

            # Scale BGM volume by -18dB (10^(-18/20) ~= 0.126)
            bgm_gain = 10.0 ** (bgm_volume_db / 20.0)
            bgm_data = bgm_data * bgm_gain

            # Apply Fade-In and Fade-Out to BGM
            fade_in_samples = min(v_len, int(fade_in_sec * sr))
            if fade_in_samples > 0:
                ramp_in = np.linspace(0.0, 1.0, fade_in_samples)[:, np.newaxis]
                bgm_data[:fade_in_samples] *= ramp_in

            fade_out_samples = min(v_len, int(fade_out_sec * sr))
            if fade_out_samples > 0:
                ramp_out = np.linspace(1.0, 0.0, fade_out_samples)[:, np.newaxis]
                bgm_data[-fade_out_samples:] *= ramp_out

            # Speech ducking: detect speech regions and drop BGM additional 4dB
            voice_rms = np.sqrt(np.mean(v_data ** 2, axis=1))
            window = int(sr * 0.1)
            if window > 0 and len(voice_rms) > window:
                smooth_rms = np.convolve(voice_rms, np.ones(window) / window, mode="same")
                duck_mask = np.where(smooth_rms > 0.03, 0.65, 1.0)[:, np.newaxis]
                bgm_data *= duck_mask

            # Mix voice and BGM
            mixed = v_data + bgm_data

            # Final peak limiting
            max_peak = np.max(np.abs(mixed))
            if max_peak > 0.98:
                mixed = mixed * (0.98 / max_peak)

            sf.write(output_wav_path, mixed, sr, subtype="PCM_16")
            return True
        except Exception as e:
            print(f"⚠️ BGM mixing notice: {e}")
            shutil.copy2(voice_wav_path, output_wav_path)
            return True

    def process_audio(
        self,
        input_wav_path: str,
        output_wav_path: str,
        bgm_path: Optional[str] = None,
        apply_denoise: bool = True,
        apply_eq: bool = True,
    ) -> bool:
        """
        Full post-processing workflow:
        Denoise -> EQ Boost -> BGM Mixing -> LUFS Normalization.
        """
        temp_dir = os.path.dirname(output_wav_path)
        step1_wav = os.path.join(temp_dir, "step1_clean.wav")
        step2_wav = os.path.join(temp_dir, "step2_mixed.wav")

        try:
            data, sr = sf.read(input_wav_path)

            if apply_denoise:
                data = self.reduce_noise(data, sr)

            if apply_eq:
                data = self.boost_voice_eq(data, sr, boost_db=3.5)

            sf.write(step1_wav, data, sr, subtype="PCM_16")

            # Mix BGM if provided
            if bgm_path and os.path.exists(bgm_path):
                self.mix_background_music(step1_wav, bgm_path, step2_wav)
                curr_in = step2_wav
            else:
                curr_in = step1_wav

            # Final LUFS normalization
            success = self.normalize_lufs_ffmpeg(curr_in, output_wav_path)
            return success
        finally:
            for p in [step1_wav, step2_wav]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass


if __name__ == "__main__":
    enhancer = AudioEnhancer()
    print("AudioEnhancer initialized successfully.")
