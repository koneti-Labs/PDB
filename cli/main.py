"""
cli/main.py — pdb entry point

Registered in pyproject.toml as:
    [project.scripts]
    pdb = "cli.main:main"

Subcommands:
    pdb listen                     # Phase 1: record + detect language + transcript
    pdb listen --device list       # list audio input devices
    pdb listen --device 2          # use device index 2
    pdb bridge                     # Phase 2: two-way patient <-> doctor translation
    pdb bridge --device 2          # bridge on specific device
"""
from __future__ import annotations

import argparse
import sys

import sounddevice as sd
from rich.console import Console
from rich.rule import Rule

from audio.handler import AudioHandler
from audio.language_id import format_language_result
from audio.recorder import Recorder
from config.languages import CONFIDENCE_THRESHOLD
from core.session import Session

console = Console()


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_listen(args: argparse.Namespace) -> None:
    """Main listen loop: record → transcribe → display → repeat."""

    # ---------------------------------------------------------------- device list
    if getattr(args, "device", None) == "list":
        console.print(sd.query_devices())
        return

    device: int | None = None
    if getattr(args, "device", None) is not None:
        try:
            device = int(args.device)
        except ValueError:
            console.print("[red]--device must be an integer index or 'list'[/red]")
            sys.exit(1)

    recorder = Recorder()
    handler = AudioHandler()
    session = Session()

    console.print()
    console.print(Rule("[bold cyan]PatientDoctorBridge — Phase 1[/bold cyan]"))
    console.print(
        "[dim]Voice capture + language detection  |  "
        "Audio deleted immediately after transcription.[/dim]"
    )

    # Preload Whisper so first transcription is not a cold start.
    try:
        handler._ensure_model()
    except Exception as exc:
        console.print(f"[yellow]Whisper preload skipped: {exc}[/yellow]")

    try:
        while True:
            # -------------------------------------------------------- record
            audio_path = recorder.record(device=device)
            if audio_path is None:
                # Bad recording (too short, device error) — let user retry
                _prompt_retry()
                continue

            # -------------------------------------------------------- transcribe + delete
            console.print("[cyan]Transcribing…[/cyan]")
            try:
                result = handler.transcribe(audio_path)
            except Exception as exc:
                console.print(f"[red]Transcription error: {exc}[/red]")
                _prompt_retry()
                continue

            # -------------------------------------------------------- store in session
            session.append("transcripts", result)

            # -------------------------------------------------------- display
            console.print()
            console.print(Rule())

            lang_label = format_language_result(result["language"], result["confidence"])
            low_conf = result["confidence"] < CONFIDENCE_THRESHOLD

            console.print(
                f"[bold]Detected language:[/bold]  {lang_label}"
                + ("  [yellow]⚠ low confidence[/yellow]" if low_conf else "")
            )
            console.print(f"[bold]Transcript:[/bold]  {result['text']}")
            console.print(Rule())
            console.print()

            # -------------------------------------------------------- continue?
            if not _ask_continue():
                break

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")

    finally:
        session.end_session()
        console.print("[dim]Session ended. No audio retained.[/dim]")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _ask_continue() -> bool:
    """Prompt the user to record again.  Returns False on EOF / Ctrl-C."""
    try:
        ans = input("Record again? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans in ("y", "yes")


def _prompt_retry() -> None:
    """Ask whether to retry after a failed recording."""
    try:
        ans = input("Try again? [Y/n] ").strip().lower()
        if ans in ("n", "no"):
            sys.exit(0)
    except (KeyboardInterrupt, EOFError):
        sys.exit(0)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdb",
        description="PatientDoctorBridge — voice capture, language detection, and translation",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # pdb listen (Phase 1)
    listen_p = sub.add_parser(
        "listen",
        help="Record from mic, detect language, print transcript (Phase 1)",
    )
    listen_p.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="Input device index (integer) or 'list' to print available devices",
    )
    listen_p.add_argument(
        "--duration",
        type=int,
        default=None,
        metavar="N",
        help="Auto-stop recording after N seconds (non-interactive testing)",
    )

    # pdb bridge (Phase 2)
    bridge_p = sub.add_parser(
        "bridge",
        help="Two-way patient <-> doctor translation via Gemma 4 (Phase 2)",
    )
    bridge_p.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="Input device index (integer) or 'list' to print available devices",
    )

    # pdb triage (Phase 3)
    triage_p = sub.add_parser(
        "triage",
        help="Emergency triage extraction from patient speech via Gemma 4 (Phase 3)",
    )
    triage_p.add_argument(
        "--device",
        metavar="DEVICE",
        default=None,
        help="Input device index (integer) or 'list' to print available devices",
    )

    # pdb prescription (Phase 4)
    prescription_p = sub.add_parser(
        "prescription",
        help="OCR a prescription image via Gemma 4 multimodal vision (Phase 4)",
    )
    prescription_p.add_argument(
        "--image",
        metavar="IMAGE",
        required=True,
        help="Path to prescription image (JPEG or PNG)",
    )

    # pdb reassure (Phase 5)
    sub.add_parser(
        "reassure",
        help="Translate emergency comfort phrases to patient language (Phase 5)",
    )

    # pdb server (Phase 6)
    server_p = sub.add_parser(
        "server",
        help="Start the web UI server (Phase 6)",
    )
    server_p.add_argument(
        "--host",
        default=None,
        help="Host to bind (default: from config, 127.0.0.1)",
    )
    server_p.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to listen on (default: from config, 5000)",
    )
    server_p.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "listen":
        cmd_listen(args)
    elif args.command == "bridge":
        from cli.bridge import cmd_bridge
        cmd_bridge(args)
    elif args.command == "triage":
        from cli.triage import cmd_triage
        cmd_triage(args)
    elif args.command == "prescription":
        from cli.prescription import cmd_prescription
        cmd_prescription(args)
    elif args.command == "reassure":
        from cli.reassure import cmd_reassure
        cmd_reassure(args)
    elif args.command == "server":
        from web.server import run_server
        run_server(
            host=args.host,
            port=args.port,
            debug=getattr(args, "debug", False),
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
