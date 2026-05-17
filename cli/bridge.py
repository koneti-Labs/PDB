"""
cli/bridge.py

cmd_bridge() — the Phase 2 entry point for pdb bridge.

Full two-way exchange loop:
  1. [Patient turn]  Record Indic speech -> Whisper transcribes + detects language
                     -> Gemma 4 translates to English for doctor
  2. [Doctor turn]   Record English speech -> Whisper transcribes (English locked)
                     -> Gemma 4 translates back to patient's detected language
  3. Session stores both turns; end_session() clears everything on exit.

Privacy contract inherited from Phase 1:
  - Audio deleted immediately after each transcribe() call (os.unlink in finally).
  - Session data is memory-only; wiped on end_session().
"""
from __future__ import annotations

import argparse
import sys

import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from audio.handler import AudioHandler, TranscriptResult
from audio.language_id import format_language_result
from audio.recorder import Recorder
from config.languages import CONFIDENCE_THRESHOLD, LANGUAGE_DISPLAY
from core.engine import GemmaEngine
from core.session import Session
from translation.service import TranslationService

console = Console()


# ---------------------------------------------------------------------------
# Main command handler
# ---------------------------------------------------------------------------

def cmd_bridge(args: argparse.Namespace) -> None:
    """Run the two-way patient <-> doctor bridge."""

    device: int | None = _resolve_device(args)
    if device == "list":
        console.print(sd.query_devices())
        return

    engine = GemmaEngine()
    service = TranslationService(engine)
    recorder = Recorder()
    handler = AudioHandler()
    session = Session()

    console.print()
    console.print(Rule("[bold cyan]PatientDoctorBridge — Phase 2[/bold cyan]"))
    console.print(
        "[dim]Privacy: audio deleted immediately after each transcription.[/dim]\n"
    )

    # Check Ollama connectivity once at startup, then warm up the models so
    # the first user-facing translation doesn't pay a cold-start penalty.
    # warmup() is idempotent — if another phase already warmed the models in
    # this process, it skips the network round-trips automatically.
    _check_ollama(engine)
    console.print("[dim]Loading Gemma 4 models into VRAM...[/dim]")
    engine.warmup()
    # Whisper warmup -- load the model into RAM up-front too.
    try:
        handler._ensure_model()
    except Exception as exc:
        console.print(f"[yellow]Whisper preload skipped: {exc}[/yellow]")

    try:
        while True:
            # ============================================================
            # TURN 1 — Patient speaks
            # ============================================================
            console.print(Rule("[blue]Patient Turn[/blue]"))

            patient_result = _record_and_transcribe(
                recorder, handler, device, label="patient"
            )
            if patient_result is None:
                if not _ask_retry("Patient recording failed. Try again?"):
                    break
                continue

            # Display transcript
            _display_transcript(patient_result, role="patient")

            # Translate patient → doctor
            console.print("[cyan]Translating for doctor (Gemma 4)...[/cyan]")
            try:
                for_doctor = service.patient_to_doctor(
                    patient_result["text"], patient_result["language"]
                )
            except RuntimeError as exc:
                console.print(f"[red]Translation error: {exc}[/red]")
                if not _ask_retry("Retry from patient turn?"):
                    break
                continue

            console.print(
                Panel(
                    Text(for_doctor, style="bold white"),
                    title="[green]For Doctor[/green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

            # Store patient turn in session
            session.append("exchanges", {
                "role": "patient",
                "language": patient_result["language"],
                "original": patient_result["text"],
                "translated": for_doctor,
            })

            # ============================================================
            # TURN 2 — Doctor responds
            # ============================================================
            console.print()
            console.print(Rule("[green]Doctor Turn[/green]"))

            doctor_result = _record_and_transcribe(
                recorder, handler, device,
                label="doctor",
                locked_language="en",   # Doctor speaks English
            )
            if doctor_result is None:
                console.print("[yellow]Skipping doctor turn.[/yellow]")
            else:
                # Display doctor transcript
                console.print(
                    Panel(
                        Text(doctor_result["text"], style="bold white"),
                        title="[green]Doctor Said (English)[/green]",
                        border_style="green",
                        padding=(1, 2),
                    )
                )

                # Translate doctor → patient (use patient's detected language)
                pat_lang = patient_result['language']
                lang_disp = LANGUAGE_DISPLAY.get(pat_lang, pat_lang)
                console.print(
                    f"[cyan]Translating for patient "
                    f"({lang_disp}) (Gemma 4)...[/cyan]"
                )
                try:
                    for_patient = service.doctor_to_patient(
                        doctor_result["text"], patient_result["language"]
                    )
                except RuntimeError as exc:
                    console.print(f"[red]Translation error: {exc}[/red]")
                    for_patient = "[translation failed]"

                lang_display = LANGUAGE_DISPLAY.get(patient_result["language"], "")
                console.print(
                    Panel(
                        Text(for_patient, style="bold white"),
                        title=f"[blue]For Patient ({lang_display})[/blue]",
                        border_style="blue",
                        padding=(1, 2),
                    )
                )

                # Read the translation aloud so the patient can hear it
                _speak_text(for_patient, patient_result["language"])

                # Store doctor turn in session
                session.append("exchanges", {
                    "role": "doctor",
                    "language": "en",
                    "original": doctor_result["text"],
                    "translated": for_patient,
                    "target_language": patient_result["language"],
                })

            # ============================================================
            # Continue?
            # ============================================================
            console.print()
            if not _ask_continue():
                break

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")

    finally:
        session.end_session()
        console.print("[dim]Session ended. No audio or text retained.[/dim]")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve_device(args: argparse.Namespace) -> int | None | str:
    """Return device index, None (default), or 'list'."""
    raw = getattr(args, "device", None)
    if raw == "list":
        return "list"
    if raw is not None:
        try:
            return int(raw)
        except ValueError:
            console.print("[red]--device must be an integer index or 'list'[/red]")
            sys.exit(1)
    return None


def _check_ollama(engine: GemmaEngine) -> None:
    """Print a quick connectivity check; warn but don't abort if models missing."""
    console.print("[dim]Checking Ollama connectivity...[/dim]", end=" ")
    try:
        status = engine.check_connectivity()
        all_ok = all(status.values())
        if all_ok:
            console.print("[green]OK[/green]")
        else:
            console.print("[yellow]WARNING[/yellow]")
            for model, available in status.items():
                mark = "[green]OK[/green]" if available else "[red]MISSING[/red]"
                console.print(f"  {model}: {mark}")
            console.print(
                "[yellow]Run [bold]ollama pull <model>[/bold] "
                "to download missing models.[/yellow]"
            )
    except Exception as exc:
        console.print(f"[red]FAILED ({exc})[/red]")
        console.print(
            "[yellow]Make sure Ollama is running: [bold]ollama serve[/bold][/yellow]"
        )


def _record_and_transcribe(
    recorder: Recorder,
    handler: AudioHandler,
    device: int | None,
    label: str,
    locked_language: str | None = None,
) -> TranscriptResult | None:
    """
    Record audio and transcribe it.

    Parameters
    ----------
    locked_language:
        If set (e.g. "en" for doctor turn), Whisper skips language detection
        and transcribes directly in that language.
    """
    audio_path = recorder.record(device=device)
    if audio_path is None:
        return None

    console.print(f"[cyan]Transcribing {label} audio...[/cyan]")
    try:
        if locked_language:
            result = handler.transcribe_locked(audio_path, language=locked_language)
        else:
            result = handler.transcribe(audio_path)
        return result
    except Exception as exc:
        console.print(f"[red]Transcription error: {exc}[/red]")
        return None


def _display_transcript(result: TranscriptResult, role: str) -> None:
    """Render transcript panel for patient turn."""
    lang_label = format_language_result(result["language"], result["confidence"])
    low_conf = result["confidence"] < CONFIDENCE_THRESHOLD

    console.print(
        Panel(
            Text.assemble(
                (f"Language:   {lang_label}", "dim"),
                ("\n" + ("  ⚠ low confidence" if low_conf else ""), "yellow" if low_conf else ""),
                (f"\n\nTranscript: {result['text']}", "bold white"),
            ),
            title=f"[blue]{role.capitalize()} Said[/blue]",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _ask_continue() -> bool:
    """Ask whether to start a new exchange. Returns False on EOF / Ctrl-C."""
    try:
        ans = input("Another exchange? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans in ("y", "yes")

def _ask_retry(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans not in ("n", "no")


def _speak_text(text: str, lang_code: str = "en") -> None:
    """
    Read *text* aloud using system TTS so the patient can hear the translation.

    Tries three backends in order:
      1. pyttsx3  -- if installed (offline, cross-platform)
      2. Windows SAPI via PowerShell  -- Windows only, no extra install
      3. macOS `say` / Linux `espeak-ng`

    Runs asynchronously (non-blocking) so the CLI does not freeze while
    speaking.  Failures are swallowed and printed as dim notices only.
    """
    import platform
    import subprocess
    import threading

    if not text.strip():
        return

    def _run() -> None:
        # Backend 1: pyttsx3 (optional install, best cross-platform quality)
        try:
            import pyttsx3  # type: ignore[import]
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text)
            engine.runAndWait()
            return
        except Exception:
            pass

        # Backend 2+3: system TTS via subprocess
        try:
            system = platform.system()
            if system == "Windows":
                # PowerShell SAPI -- always available on Windows 7+
                safe = text.replace("'", "\'").replace('"', '\"').replace("\n", " ")
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f"$s.Speak('{safe}')"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                    timeout=60,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            elif system == "Darwin":
                subprocess.run(["say", text], timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                subprocess.run(["espeak-ng", text], timeout=60,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            console.print(f"[dim]TTS unavailable: {exc}[/dim]")

    # Fire-and-forget -- don't block the main CLI loop
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    console.print("[dim]  \U0001f50a Reading translation aloud…[/dim]")
