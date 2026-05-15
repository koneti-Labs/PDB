"""
cli/prescription.py

cmd_prescription() -- Phase 4 entry point for pdb prescription.

Prescription OCR loop:
  1. User provides --image <path> to a JPEG/PNG prescription.
  2. GemmaEngine.transcribe_prescription() (gemma4:e4b, multimodal) extracts
     structured JSON: medicines list, doctor_name, patient_name, date, notes.
  3. PrescriptionService parses + validates the JSON into a PrescriptionResult.
  4. Rich renders a prescription card with a table of medicines.

Privacy: image is read once and never stored; no data leaves the device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from core.engine import GemmaEngine
from translation.prescription import PrescriptionResult, PrescriptionService

console = Console()


def cmd_prescription(args: argparse.Namespace) -> None:
    """Run prescription OCR on a supplied image file."""

    image_path = getattr(args, "image", None)
    if not image_path:
        console.print("[red]Error: --image <path> is required.[/red]")
        sys.exit(1)

    path = Path(image_path)
    if not path.exists():
        console.print(f"[red]Error: file not found: {path}[/red]")
        sys.exit(1)

    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        console.print("[yellow]Warning: file may not be a supported image type (JPEG/PNG recommended).[/yellow]")

    engine = GemmaEngine()
    service = PrescriptionService(engine)

    console.print()
    console.print(Rule("[bold cyan]PatientDoctorBridge — Phase 4: Prescription OCR[/bold cyan]"))
    console.print("[dim]Extracting medicine information via Gemma 4 multimodal vision.[/dim]\n")

    _check_ollama(engine)
    console.print("[dim]Loading Gemma 4 models into VRAM...[/dim]")
    engine.warmup()

    console.print(f"[cyan]Processing image: {path.name} …[/cyan]")
    try:
        result = service.extract(str(path))
    except ValueError as exc:
        console.print(f"[red]Parse error: {exc}[/red]")
        sys.exit(1)
    except RuntimeError as exc:
        console.print(f"[red]Ollama error: {exc}[/red]")
        sys.exit(1)

    _display_prescription_card(result, path.name)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_prescription_card(result: PrescriptionResult, filename: str) -> None:
    """Render a rich prescription card."""

    # ── Header info ──────────────────────────────────────────────────────────
    meta_table = Table(show_header=False, box=None, padding=(0, 1))
    meta_table.add_column("Field", style="bold dim", width=18)
    meta_table.add_column("Value", style="white")

    meta_table.add_row("File", filename)
    meta_table.add_row("Doctor", result["doctor_name"])
    meta_table.add_row("Patient", result["patient_name"])
    meta_table.add_row("Date", result["date"])
    if result["notes"]:
        meta_table.add_row("Notes", result["notes"])

    console.print(Panel(meta_table, title="[bold cyan]Prescription Details[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))

    # ── Medicines table ──────────────────────────────────────────────────────
    medicines = result["medicines"]
    if not medicines:
        console.print("[yellow]No medicines found in the prescription.[/yellow]")
        return

    med_table = Table(
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )
    med_table.add_column("#", style="dim", width=3)
    med_table.add_column("Medicine", style="bold white", min_width=18)
    med_table.add_column("Dosage", style="white", min_width=10)
    med_table.add_column("Form", style="dim white", min_width=8)
    med_table.add_column("Frequency", style="white", min_width=14)
    med_table.add_column("Duration", style="white", min_width=10)
    med_table.add_column("Instructions", style="dim white", min_width=16)

    for i, med in enumerate(medicines, 1):
        med_table.add_row(
            str(i),
            med["name"],
            med["dosage"],
            med["form"],
            med["frequency"],
            med["duration"],
            med["instructions"],
        )

    console.print(Panel(med_table, title=f"[bold cyan]Medicines ({len(medicines)})[/bold cyan]",
                        border_style="cyan", padding=(1, 2)))
    console.print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
                "to download the vision model.[/yellow]"
            )
    except Exception as exc:
        console.print(f"[red]FAILED ({exc})[/red]")
        console.print("[yellow]Make sure Ollama is running: [bold]ollama serve[/bold][/yellow]")
    console.print()
