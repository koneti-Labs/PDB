"""
audio/preprocessor.py

Audio quality analysis and preprocessing before Whisper transcription.

This module provides:
1. Signal-to-Noise Ratio (SNR) estimation
2. Silence detection and trimming
3. Audio normalization
4. Quality metrics for debugging

The goal is to improve transcription accuracy by ensuring clean audio
reaches Whisper, especially for Hindi speech which can be misclassified
as Urdu when audio quality is poor.
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np
from rich.console import Console

console = Console()


class AudioQualityMetrics(TypedDict):
    snr_db: float
    silence_ratio: float
    peak_amplitude: float
    rms_level: float
    is_acceptable: bool
    warnings: list[str]


class AudioPreprocessor:
    """
    Analyze and preprocess audio before transcription.

    Quality thresholds:
    - SNR > 10 dB: Good quality
    - SNR 5-10 dB: Acceptable
    - SNR < 5 dB: Poor quality (warning)
    - Silence ratio > 0.7: Too much silence (warning)
    """

    MIN_SNR_DB = 5.0
    MAX_SILENCE_RATIO = 0.7
    SILENCE_THRESHOLD = 0.01  # Amplitude threshold for silence detection
    TARGET_RMS = 0.1  # Target RMS for normalization

    def __init__(self):
        pass

    def analyze_quality(self, audio: np.ndarray, sample_rate: int = 16000) -> AudioQualityMetrics:
        """
        Analyze audio quality and return metrics.

        Parameters
        ----------
        audio:
            Audio samples as numpy array (int16 or float32)
        sample_rate:
            Sample rate in Hz (default: 16000)

        Returns
        -------
        AudioQualityMetrics with SNR, silence ratio, and quality flags
        """
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Ensure mono
        if len(audio_float.shape) > 1:
            audio_float = audio_float.mean(axis=1)

        warnings = []

        # Calculate RMS level
        rms = np.sqrt(np.mean(audio_float ** 2))

        # Calculate peak amplitude
        peak = np.max(np.abs(audio_float))

        # Detect silence
        silence_mask = np.abs(audio_float) < self.SILENCE_THRESHOLD
        silence_ratio = np.sum(silence_mask) / len(audio_float)

        # Estimate SNR
        snr_db = self._estimate_snr(audio_float)

        # Quality checks
        is_acceptable = True

        if snr_db < self.MIN_SNR_DB:
            warnings.append(f"Low SNR ({snr_db:.1f} dB) - noisy audio may affect accuracy")
            is_acceptable = False

        if silence_ratio > self.MAX_SILENCE_RATIO:
            warnings.append(
                f"High silence ratio ({silence_ratio:.1%}) - recording may be too quiet"
            )

        if rms < 0.01:
            warnings.append(f"Very low RMS level ({rms:.4f}) - audio may be too quiet")
            is_acceptable = False

        if peak > 0.95:
            warnings.append(f"Peak amplitude near clipping ({peak:.2f}) - audio may be distorted")

        return AudioQualityMetrics(
            snr_db=snr_db,
            silence_ratio=silence_ratio,
            peak_amplitude=peak,
            rms_level=rms,
            is_acceptable=is_acceptable,
            warnings=warnings,
        )

    def preprocess(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Preprocess audio for better transcription quality.

        Steps:
        1. Convert to float32
        2. Trim leading/trailing silence
        3. Normalize RMS level
        4. Apply gentle noise gate

        Parameters
        ----------
        audio:
            Audio samples as numpy array (int16 or float32)
        sample_rate:
            Sample rate in Hz (default: 16000)

        Returns
        -------
        Preprocessed audio as float32 numpy array
        """
        # Convert to float if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Ensure mono
        if len(audio_float.shape) > 1:
            audio_float = audio_float.mean(axis=1)

        # Trim silence from start and end
        audio_float = self._trim_silence(audio_float)

        # Apply gentle noise gate (reduce very quiet sections)
        audio_float = self._apply_noise_gate(audio_float)

        # Normalize RMS level
        audio_float = self._normalize_rms(audio_float)

        return audio_float

    def _estimate_snr(self, audio: np.ndarray) -> float:
        """
        Estimate Signal-to-Noise Ratio in dB.

        Uses a simple heuristic:
        - Signal: top 50% of energy frames
        - Noise: bottom 20% of energy frames
        """
        # Calculate frame energy (100ms frames)
        frame_length = 1600  # 100ms at 16kHz
        num_frames = len(audio) // frame_length

        if num_frames < 5:
            # Too short for reliable SNR estimation
            return 15.0  # Assume reasonable quality

        frame_energies = []
        for i in range(num_frames):
            frame = audio[i * frame_length:(i + 1) * frame_length]
            energy = np.sum(frame ** 2)
            frame_energies.append(energy)

        frame_energies = np.array(frame_energies)

        # Signal: top 50% of frames
        signal_threshold = np.percentile(frame_energies, 50)
        signal_frames = frame_energies[frame_energies >= signal_threshold]
        signal_power = np.mean(signal_frames) if len(signal_frames) > 0 else 1e-10

        # Noise: bottom 20% of frames
        noise_threshold = np.percentile(frame_energies, 20)
        noise_frames = frame_energies[frame_energies <= noise_threshold]
        noise_power = np.mean(noise_frames) if len(noise_frames) > 0 else 1e-10

        # Avoid division by zero
        if noise_power < 1e-10:
            noise_power = 1e-10

        snr = signal_power / noise_power
        snr_db = 10 * np.log10(snr) if snr > 0 else 0.0

        return snr_db

    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Trim leading and trailing silence.

        Parameters
        ----------
        audio:
            Audio samples (float32)
        threshold:
            Amplitude threshold for silence detection

        Returns
        -------
        Trimmed audio
        """
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold

        if not np.any(non_silent):
            # All silence - return original
            return audio

        # Find first and last non-silent sample
        non_silent_indices = np.where(non_silent)[0]
        start = max(0, non_silent_indices[0] - 800)  # Keep 50ms before
        end = min(len(audio), non_silent_indices[-1] + 800)  # Keep 50ms after

        return audio[start:end]

    def _apply_noise_gate(
        self, audio: np.ndarray, threshold: float = 0.005, ratio: float = 0.5
    ) -> np.ndarray:
        """
        Apply a gentle noise gate to reduce background noise.

        Parameters
        ----------
        audio:
            Audio samples (float32)
        threshold:
            Amplitude threshold below which to apply gating
        ratio:
            Reduction ratio (0.5 = reduce by 50%)

        Returns
        -------
        Gated audio
        """
        # Create a copy
        gated = audio.copy()

        # Apply gate to quiet samples
        quiet_mask = np.abs(gated) < threshold
        gated[quiet_mask] *= ratio

        return gated

    def _normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """
        Normalize audio to target RMS level.

        Parameters
        ----------
        audio:
            Audio samples (float32)
        target_rms:
            Target RMS level (0.1 is a good default)

        Returns
        -------
        Normalized audio
        """
        current_rms = np.sqrt(np.mean(audio ** 2))

        if current_rms < 1e-6:
            # Avoid division by zero for silent audio
            return audio

        gain = target_rms / current_rms

        # Limit gain to avoid excessive amplification
        gain = min(gain, 10.0)

        normalized = audio * gain

        # Soft clip to avoid distortion
        normalized = np.tanh(normalized * 0.9) / 0.9

        return normalized

    def analyze_and_log(
        self, audio: np.ndarray, sample_rate: int = 16000, label: str = "audio"
    ) -> AudioQualityMetrics:
        """
        Analyze audio quality and log results to console.

        Parameters
        ----------
        audio:
            Audio samples
        sample_rate:
            Sample rate in Hz
        label:
            Label for logging (e.g., "patient", "doctor")

        Returns
        -------
        AudioQualityMetrics
        """
        metrics = self.analyze_quality(audio, sample_rate)

        # Log quality metrics
        console.print(f"[dim]Audio quality ({label}):[/dim]")
        console.print(f"[dim]  SNR: {metrics['snr_db']:.1f} dB[/dim]")
        console.print(f"[dim]  Silence: {metrics['silence_ratio']:.1%}[/dim]")
        console.print(f"[dim]  RMS: {metrics['rms_level']:.4f}[/dim]")
        console.print(f"[dim]  Peak: {metrics['peak_amplitude']:.2f}[/dim]")

        # Log warnings
        if metrics['warnings']:
            for warning in metrics['warnings']:
                console.print(f"[yellow]⚠ {warning}[/yellow]")
        else:
            console.print("[green]✓ Audio quality acceptable[/green]")

        return metrics
