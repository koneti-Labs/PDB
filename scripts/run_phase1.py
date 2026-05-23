import sys
import time
from pathlib import Path

# Add the parent directory to the path so we can import core
sys.path.insert(0, str(Path(__file__).parent.parent))

import keyboard
from arch.audio_handler import AudioHandler
from arch.config import SUPPORTED_LANGUAGES
from rich.console import Console
from rich.panel import Panel

console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]Patient Doctor Bridge[/bold green] - Phase 1\n"
        "[cyan]Voice Capture + Language Detection (Push-to-Talk)[/cyan]",
        title="🚀 Starting",
        border_style="green"
    ))

    console.print(f"Supported Languages: {', '.join(SUPPORTED_LANGUAGES.values())}\n")
    console.print("[bold]Instructions:[/bold]")
    console.print("• Hold [white on blue] SPACEBAR [/white on blue] while speaking")
    console.print("• Release to stop recording")
    console.print("• Type [red]'q'[/red] to quit\n")

    handler = AudioHandler()

    # Preload model for better UX
    console.print("[yellow]Preloading Whisper model...[/yellow]")
    handler.load_model()
    console.print("[green]✓ Ready![/green]\n")

    while True:
        try:
            console.print("[dim]Hold SPACEBAR to record, or press 'q' to quit[/dim]")

            # Wait for spacebar or 'q' key
            while True:
                if keyboard.is_pressed('space'):
                    break
                if keyboard.is_pressed('q'):
                    console.print("\n[yellow]Session ended by user.[/yellow]")
                    console.print(
                        "[bold green]Thank you for using"
                        " Patient Doctor Bridge![/bold green]"
                    )
                    return
                time.sleep(0.01)

            audio_file = handler.record_push_to_talk()

            if audio_file:
                result = handler.transcribe(audio_file)

                console.print(Panel(
                    f"[bold]Detected Language:[/bold] {result['language_name']} "
                    f"({result['language_code']}) - Confidence: {result['confidence']:.2%}\n\n"
                    f"[bold]Transcription:[/bold]\n{result['transcription']}",
                    title="✅ Result",
                    border_style="green"
                ))

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
