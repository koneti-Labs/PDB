"""
cli/triage.py

cmd_triage() — the Phase 3 entry point for pdb triage.

Emergency triage extraction loop:
  1. Record patient speech (any of the 5 supported languages).
  2. Whisper transcribes + detects language.
  3. GemmaEngine.emergency_triage() (gemma4:e4b, think=True) extracts
     structured JSON: chief_complaint, severity, duration, symptoms,
     vitals_mentioned, needs_immediate_attention.
  4. TriageService parses and validates the JSON into a TriageResult.
  5. Rich renders a triage card — bright RED alert if needs_immediate_attention.
  6. Session stores each result; end_session() wipes on exit.

Privacy: audio deleted immediately after transcription (handler.transcribe).
"""
from __future__ import annotations

import argparse
import sys

import sounddevice as sd
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from audio.handler import AudioHandler
from audio.language_id import format_language_result
from audio.recorder import Recorder
from config.languages import CONFIDENCE_THRESHOLD, LANGUAGE_DISPLAY
from core.engine import GemmaEngine
from core.session import Session
from translation.triage import TriageResult, TriageService

console = Console()

# Severity → (colour, label) for the triage card
_SEVERITY_STYLE: dict[str, tuple[str, str]] = {
    "critical": ("bold red on dark_red", "CRITICAL ⚠"),
    "severe":   ("bold red", "SEVERE"),
    "moderate": ("bold yellow", "MODERATE"),
    "mild":     ("bold green", "MILD"),
    "unknown":  ("bold white", "UNKNOWN"),
}


# ---------------------------------------------------------------------------
# Main command handler
# ---------------------------------------------------------------------------

def cmd_triage(args: argparse.Namespace) -> None:
    """Run the emergency triage extraction loop."""

    device: int | None = _resolve_device(args)
    if device == "list":
        console.print(sd.query_devices())
        return

    engine = GemmaEngine()
    service = TriageService(engine)
    recorder = Recorder()
    handler = AudioHandler()
    session = Session()

    console.print()
    console.print(Rule("[bold red]PatientDoctorBridge — Phase 3: Emergency Triage[/bold red]"))
    console.print(
        "[dim]Record patient speech → Gemma 4 extracts triage information locally.[/dim]\n"
        "[dim]Privacy: audio deleted immediately after transcription.[/dim]\n"
    )

    _check_ollama(engine)
    console.print("[dim]Loading Gemma 4 models into VRAM...[/dim]")
    engine.warmup()
    try:
        handler._ensure_model()  # preload Whisper too
    except Exception as exc:
        console.print(f"[yellow]Whisper preload skipped: {exc}[/yellow]")

    try:
        while True:
            console.print(Rule("[cyan]New Patient[/cyan]"))

            # ----------------------------------------------------------------
            # Step 1 — Record + transcribe
            # ----------------------------------------------------------------
            audio_path = recorder.record(device=device)
            if audio_path is None:
                if not _ask_retry("Recording failed. Try again?"):
                    break
                continue

            console.print("[cyan]Transcribing…[/cyan]")
            try:
                result = handler.transcribe(audio_path)
            except Exception as exc:
                console.print(f"[red]Transcription error: {exc}[/red]")
                if not _ask_retry("Try again?"):
                    break
                continue

            # Show transcript
            lang_label = format_language_result(result["language"], result["confidence"])
            low_conf = result["confidence"] < CONFIDENCE_THRESHOLD
            console.print(
                f"  Detected: [bold]{lang_label}[/bold]"
                + ("  [yellow]⚠ low confidence — results may be imprecise[/yellow]" if low_conf else "")
            )
            console.print(f"  Transcript: [italic]{result['text']}[/italic]\n")

            # ----------------------------------------------------------------
            # Step 2 — Triage extraction (Gemma 4)
            # ----------------------------------------------------------------
            console.print("[cyan]Extracting triage data (Gemma 4 — thinking)…[/cyan]")
            try:
                triage = service.extract(result["text"], result["language"])
            except ValueError as exc:
                console.print(f"[red]Triage parse error: {exc}[/red]")
                if not _ask_retry("Try again?"):
                    break
                continue
            except RuntimeError as exc:
                console.print(f"[red]Ollama error: {exc}[/red]")
                if not _ask_retry("Try again?"):
                    break
                continue

            # ----------------------------------------------------------------
            # Step 3 — Display triage card
            # ----------------------------------------------------------------
            _display_triage_card(triage, result["language"])

            # ----------------------------------------------------------------
            # Store in session
            # ----------------------------------------------------------------
            session.append("triage_results", {
                "transcript": result["text"],
                "language": result["language"],
                "triage": dict(triage),
            })

            # ----------------------------------------------------------------
            # Continue?
            # ----------------------------------------------------------------
            console.print()
            if not _ask_continue():
                break

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")

    finally:
        session.end_session()
        console.print("[dim]Session ended. No audio or data retained.[/dim]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_triage_card(triage: TriageResult, lang_code: str) -> None:
    """Render a rich triage card.  Red banner if needs_immediate_attention."""

    severity = triage["severity"]
    sev_style, sev_label = _SEVERITY_STYLE.get(severity, _SEVERITY_STYLE["unknown"])
    immediate = triage["needs_immediate_attention"]

    # ── Immediate-attention banner ───────────────────────────────────────────
    if immediate:
        console.print(
            Panel(
                Text("⚠  IMMEDIATE ATTENTION REQUIRED  ⚠", style="bold white", justify="center"),
                style="bold red",
                padding=(0, 2),
            )
        )

    # ── Main triage table ────────────────────────────────────────────────────
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Field", style="bold dim", width=22)
    table.add_column("Value", style="white")

    table.add_row("Chief Complaint", triage["chief_complaint"])
    table.add_row("Severity", Text(sev_label, style=sev_style))
    table.add_row("Duration", triage["duration"])
    table.add_row(
        "Symptoms",
        "\n".join(f"• {s}" for s in triage["symptoms"]) if triage["symptoms"] else "[dim]none reported[/dim]",
    )
    table.add_row(
        "Vitals Mentioned",
        "\n".join(f"• {v}" for v in triage["vitals_mentioned"])
        if triage["vitals_mentioned"]
        else "[dim]none[/dim]",
    )
    lang_display = LANGUAGE_DISPLAY.get(lang_code, lang_code)
    table.add_row("Patient Language", lang_display)

    border = "red" if immediate else "cyan"
    title_text = (
        "[bold red]TRIAGE CARD — URGENT[/bold red]"
        if immediate
        else "[bold cyan]Triage Card[/bold cyan]"
    )
    console.print(Panel(table, title=title_text, border_style=border, padding=(1, 2)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(args: argparse.Namespace) -> int | None | str:
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
                "[yellow]Run [bold]ollama pull gemma4:e4b[/bold] "
                "to download the triage model.[/yellow]"
            )
    except Exception as exc:
        console.print(f"[red]FAILED ({exc})[/red]")
        console.print(
            "[yellow]Make sure Ollama is running: [bold]ollama serve[/bold][/yellow]"
        )
    console.print()


def _ask_retry(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans not in ("n", "no")


def _ask_continue() -> bool:
    try:
        ans = input("Triage another patient? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans in ("y", "yes")
