"""
audio/recorder.py

Recorder — microphone capture via sounddevice.

Interaction model: Enter to start recording, Enter again to stop.
Audio is written to a tempfile.NamedTemporaryFile in the OS temp dir;
the path is returned to the caller.  The caller (AudioHandler.transcribe)
is responsible for deleting the file immediately after use.

No audio is written anywhere outside AUDIO_TEMP_DIR.
No audio path is logged above DEBUG level.
"""
from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from rich.console import Console
from scipy.io.wavfile import write as wav_write

from config.settings import (
    AUDIO_TEMP_DIR,
    CHANNELS,
    MAX_RECORDING_SECONDS,
    MIN_RECORDING_SECONDS,
    SAMPLE_RATE,
)

console = Console()


class Recorder:
    """
    Push-to-talk recorder: Enter → record → Enter → stop.

    Parameters
    ----------
    device:
        sounddevice device index.  None = system default.
        Pass ``"list"`` to print available devices instead of recording.
    """

    def record(self, device: int | None = None) -> Path | None:
        """
        Block until the user presses Enter, record until Enter again
        (or MAX_RECORDING_SECONDS), then return the path to a temp WAV file.

        Returns None if:
          • The audio device cannot be opened.
          • The recording is shorter than MIN_RECORDING_SECONDS.
          • No audio was captured (empty buffer).
        """
        audio_chunks: list[np.ndarray] = []
        recording_active = threading.Event()

        def _callback(
            indata: np.ndarray,
            frames: int,
            time_info: object,
            status: sd.CallbackFlags,
        ) -> None:
            if status:
                console.print(f"[yellow dim]⚠  Audio stream: {status}[/yellow dim]")
            if recording_active.is_set():
                audio_chunks.append(indata.copy())

        # ------------------------------------------------------------------ open stream
        console.print("\n[bold]Press [cyan]ENTER[/cyan] to start recording…[/bold]")
        input()

        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                callback=_callback,
                device=device,
            )
        except Exception as exc:
            console.print(f"[red]✗ Could not open audio device:[/red] {exc}")
            console.print(
                "[dim]Run [bold]pdb listen --device list[/bold] to see available devices.[/dim]"
            )
            return None

        # ------------------------------------------------------------------ record
        with stream:
            recording_active.set()
            start = time.monotonic()

            console.print(
                "[bold red]● Recording…[/bold red]"
                "  Press [cyan]ENTER[/cyan] to stop"
                f"  (max {MAX_RECORDING_SECONDS}s)"
            )

            stop_event = threading.Event()

            def _wait_for_stop() -> None:
                input()
                stop_event.set()

            stopper = threading.Thread(target=_wait_for_stop, daemon=True)
            stopper.start()

            while not stop_event.is_set():
                elapsed = time.monotonic() - start
                if elapsed >= MAX_RECORDING_SECONDS:
                    console.print(
                        f"[yellow]⚠  Max recording time "
                        f"({MAX_RECORDING_SECONDS}s) reached.[/yellow]"
                    )
                    break
                time.sleep(0.05)

            recording_active.clear()
            elapsed = time.monotonic() - start

        # ------------------------------------------------------------------ validate
        if elapsed < MIN_RECORDING_SECONDS:
            console.print(
                f"[yellow]Recording too short ({elapsed:.1f}s). "
                f"Minimum is {MIN_RECORDING_SECONDS}s — please try again.[/yellow]"
            )
            return None

        if not audio_chunks:
            console.print("[red]No audio captured.[/red]")
            return None

        # ------------------------------------------------------------------ save to temp
        audio = np.concatenate(audio_chunks, axis=0)

        tmp = tempfile.NamedTemporaryFile(
            suffix=".wav",
            dir=AUDIO_TEMP_DIR,
            delete=False,
        )
        wav_write(tmp.name, SAMPLE_RATE, audio)
        tmp.close()

        console.print(f"[dim]Captured {elapsed:.1f}s of audio.[/dim]")
        return Path(tmp.name)
