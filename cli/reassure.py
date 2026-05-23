"""
cli/reassure.py

cmd_reassure() -- Phase 5 entry point for pdb reassure.

Emergency reassurance loop:
  1. Display numbered menu of built-in phrases (URGENT / MEDICAL / COMFORT / INFO).
  2. User picks a phrase by number (or types a custom phrase).
  3. User enters the patient's language code.
  4. GemmaEngine.emergency_reassurance() (gemma4:e2b) translates the phrase.
  5. Display the translated phrase prominently.
  6. Option to translate another phrase.

Privacy: no audio, no PII stored; session ends cleanly.
"""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from config.languages import LANGUAGE_DISPLAY
from core.engine import GemmaEngine
from translation.reassurance import REASSURANCE_PHRASES, ReassuranceService

console = Console()

# Category colour map
_CAT_STYLE: dict[str, str] = {
    "URGENT":  "bold red",
    "MEDICAL": "bold yellow",
    "COMFORT": "bold green",
    "INFO":    "bold cyan",
}

_SUPPORTED_LANGS: list[str] = ["hi", "te", "kn", "ta", "en"]


def cmd_reassure(args: argparse.Namespace) -> None:
    """Run the emergency reassurance translation loop."""

    engine = GemmaEngine()
    service = ReassuranceService(engine)

    console.print()
    console.print(Rule("[bold red]PatientDoctorBridge — Phase 5: Emergency Reassurance[/bold red]"))
    console.print(
        "[dim]Translate a comfort phrase into the patient's language instantly.[/dim]\n"
        "[dim]Uses Gemma 4 (gemma4:e2b) locally — nothing leaves the device.[/dim]\n"
    )

    _check_ollama(engine)
    console.print("[dim]Loading Gemma 4 into VRAM...[/dim]")
    engine.warmup()

    try:
        while True:
            # -------------------------------------------------------- phrase menu
            _display_phrase_menu()

            phrase = _pick_phrase()
            if phrase is None:
                break

            # -------------------------------------------------------- language
            lang_code = _pick_language()
            if lang_code is None:
                break

            # -------------------------------------------------------- translate
            lang_display = LANGUAGE_DISPLAY.get(lang_code, lang_code)
            if lang_code == "en":
                translated = phrase
                console.print("\n[dim]Patient language is English — no translation needed.[/dim]")
            else:
                console.print(f"\n[cyan]Translating to {lang_display} (Gemma 4)…[/cyan]")
                try:
                    translated = service.translate(phrase, lang_code)
                except RuntimeError as exc:
                    console.print(f"[red]Ollama error: {exc}[/red]")
                    if not _ask_retry("Try again?"):
                        break
                    continue

            # -------------------------------------------------------- display
            _display_translation(phrase, translated, lang_display)

            # -------------------------------------------------------- continue?
            console.print()
            if not _ask_continue():
                break

    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")

    console.print("[dim]Session ended.[/dim]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _display_phrase_menu() -> None:
    """Print the numbered phrase menu."""
    table = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=4)
    table.add_column("Category", width=10)
    table.add_column("Phrase", style="white")

    for i, (cat, phrase) in enumerate(REASSURANCE_PHRASES, 1):
        table.add_row(
            str(i),
            Text(cat, style=_CAT_STYLE.get(cat, "white")),
            phrase,
        )

    table.add_row("C", "[dim]custom[/dim]", "[dim]Enter your own phrase[/dim]")

    console.print(Panel(table, title="[bold]Reassurance Phrases[/bold]",
                        border_style="cyan", padding=(1, 2)))


def _display_translation(original: str, translated: str, lang_display: str) -> None:
    """Display the translated phrase in a prominent panel."""
    content = Text()
    content.append("English:\n", style="dim")
    content.append(f"  {original}\n\n", style="italic white")
    content.append(f"{lang_display}:\n", style="bold")
    content.append(f"  {translated}", style="bold white")

    console.print(Panel(content, title=f"[bold green]Translation — {lang_display}[/bold green]",
                        border_style="green", padding=(1, 2)))


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _pick_phrase() -> str | None:
    """Prompt for phrase selection; returns English phrase string or None to quit."""
    phrases = REASSURANCE_PHRASES
    while True:
        try:
            raw = input(f"\nSelect phrase [1-{len(phrases)}, C=custom, Q=quit]: ").strip()
        except (KeyboardInterrupt, EOFError):
            return None

        if raw.lower() == "q":
            return None

        if raw.lower() == "c":
            try:
                custom = input("Enter custom phrase (English): ").strip()
            except (KeyboardInterrupt, EOFError):
                return None
            if custom:
                return custom
            console.print("[yellow]Empty phrase — try again.[/yellow]")
            continue

        try:
            idx = int(raw) - 1
            if 0 <= idx < len(phrases):
                return phrases[idx][1]
        except ValueError:
            pass

        console.print(f"[yellow]Enter a number between 1 and {len(phrases)}, C, or Q.[/yellow]")


def _pick_language() -> str | None:
    """Prompt for patient language code; returns ISO 639-1 code or None to quit."""
    lang_list = ", ".join(f"{k}={v}" for k, v in LANGUAGE_DISPLAY.items())
    console.print(f"\nLanguages: {lang_list}")
    while True:
        try:
            raw = input("Patient language code (or Q to quit): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None

        if raw == "q":
            return None

        if raw in _SUPPORTED_LANGS:
            return raw

        console.print(
            f"[yellow]Unrecognised code. Choose from:"
            f" {', '.join(_SUPPORTED_LANGS)}[/yellow]"
        )


def _ask_retry(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [Y/n] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans not in ("n", "no")


def _ask_continue() -> bool:
    try:
        ans = input("Translate another phrase? [y/N] ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# Ollama check
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
    except Exception as exc:
        console.print(f"[red]FAILED ({exc})[/red]")
        console.print("[yellow]Make sure Ollama is running: [bold]ollama serve[/bold][/yellow]")
    console.print()
