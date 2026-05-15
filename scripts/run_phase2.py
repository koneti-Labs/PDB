import sys
from pathlib import Path
import time
import keyboard
from rich.console import Console
from rich.panel import Panel

sys.path.insert(0, str(Path(__file__).parent.parent))

from arch.audio_handler import AudioHandler
from arch.translation_service import TranslationService
from arch.config import SUPPORTED_LANGUAGES

console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]Patient Doctor Bridge[/bold green] - Phase 2\n"
        "[cyan]Patient → Doctor Translation (Gemma 4)[/cyan]",
        title="🚀",
        border_style="green"
    ))

    audio_handler = AudioHandler()
    translator = TranslationService()

    console.print("[yellow]Preloading models...[/yellow]")
    audio_handler.load_model()
    console.print("[green]✓ Whisper Ready | Gemma 4 Ready[/green]\n")

    while True:
        try:
            console.print("\n[dim]Hold SPACEBAR to speak to Doctor...[/dim]")

            # Wait for input
            while True:
                if keyboard.is_pressed('q'):
                    console.print("\n[yellow]Goodbye![/yellow]")
                    return
                if keyboard.is_pressed('space'):
                    break
                time.sleep(0.01)

            # Record
            audio_file = audio_handler.record_push_to_talk()
            if not audio_file:
                continue

            # Transcribe
            result = audio_handler.transcribe(audio_file)

            console.print(Panel(
                f"[bold]Detected:[/bold] {result['language_name']} "
                f"({result['language_code']}) — {result['confidence']:.2%}\n\n"
                f"[bold]Patient Said:[/bold]\n{result['transcription']}",
                title="🎤 Patient Input",
                border_style="blue"
            ))

            # Translate using Gemma 4
            console.print("[cyan]Translating to Doctor (Gemma 4 2B)...[/cyan]")
            doctor_version = translator.patient_to_doctor(
                result['transcription'], 
                result['language_name']
            )

            console.print(Panel(
                f"[bold]Doctor's Version:[/bold]\n{doctor_version}",
                title="🩺 For Doctor",
                border_style="green"
            ))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()