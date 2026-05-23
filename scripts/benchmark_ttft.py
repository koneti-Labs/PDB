"""
scripts/benchmark_ttft.py

Benchmark Time-To-First-Token (TTFT) for both Gemma 4 models.

Usage:
    python scripts/benchmark_ttft.py
    python scripts/benchmark_ttft.py --runs 5

Target (Pi 5 + Hailo-10H NPU):  TTFT < 500 ms
Target (Kaggle T4 GPU):          TTFT < 150 ms
Target (CPU / laptop):           TTFT < 3000 ms

Results are printed as a Rich table and written to benchmark_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.hardware import HARDWARE, print_hardware_report
from core.engine import MODELS, GemmaEngine, InferenceMode

app = typer.Typer(add_completion=False)
console = Console()


def _measure_ttft(engine: GemmaEngine, mode: InferenceMode, prompt: str) -> float:
    """
    Return wall-clock seconds from the first client.generate() call until
    the full response is received.

    TTFT here is measured as end-to-end latency for a 1-token prompt.
    True streaming TTFT requires the Ollama streaming API; this is a
    practical substitute that correlates tightly with real TTFT on CPU.
    """
    t0 = time.perf_counter()
    try:
        engine.generate(prompt, mode=mode, num_ctx=64)
    except Exception as exc:
        console.print(f"[red]  ⚠ generate failed: {exc}[/red]")
        return -1.0
    return time.perf_counter() - t0


@app.command()
def benchmark(
    runs: int = typer.Option(3, help="Number of timed runs per model"),
    output: Optional[Path] = typer.Option(
        None, help="Path to write JSON results (default: benchmark_results.json)"
    ),
) -> None:
    """Benchmark Gemma 4 TTFT on the current hardware."""
    out_path = output or Path("benchmark_results.json")

    console.rule("[bold cyan]PatientDoctorBridge — TTFT Benchmark[/bold cyan]")
    print_hardware_report()
    console.print()

    engine = GemmaEngine()

    # Warm up (loads models into VRAM; not counted in benchmark)
    console.print("[dim]Warming up models (one-time)…[/dim]")
    engine.warmup()
    console.print()

    PROMPTS = {
        InferenceMode.FAST_TRANSLATION: (
            "Translate from Hindi to English: मेरे सिर में दर्द है।"
        ),
        InferenceMode.REASONING_EXTRACTION: (
            "Extract symptoms from: I have chest pain since morning."
        ),
    }

    results: dict[str, dict] = {}
    summary_table = Table(
        title=f"TTFT Results ({runs} runs each)",
        show_header=True,
        header_style="bold",
    )
    summary_table.add_column("Model",        min_width=18)
    summary_table.add_column("Mode",         min_width=20)
    summary_table.add_column("Min (ms)",     justify="right")
    summary_table.add_column("Median (ms)",  justify="right")
    summary_table.add_column("Mean (ms)",    justify="right")
    summary_table.add_column("StdDev (ms)",  justify="right")
    summary_table.add_column("Target",       justify="center")

    TARGETS_MS = {
        InferenceMode.FAST_TRANSLATION:     500,
        InferenceMode.REASONING_EXTRACTION: 3000,
    }

    for mode, prompt in PROMPTS.items():
        model_tag = MODELS[mode]
        console.print(f"[dim]Benchmarking {model_tag} ({mode.value})…[/dim]")

        timings: list[float] = []
        for i in range(runs):
            t = _measure_ttft(engine, mode, prompt)
            if t >= 0:
                timings.append(t * 1000)  # → ms
                console.print(f"  Run {i + 1}/{runs}: {t * 1000:.0f} ms")
            else:
                console.print(f"  Run {i + 1}/{runs}: FAILED")

        if not timings:
            console.print(f"[red]  No successful runs for {model_tag}[/red]")
            continue

        target_ms = TARGETS_MS[mode]
        med = median(timings)
        met = "✅" if med <= target_ms else "❌"

        summary_table.add_row(
            model_tag,
            mode.value,
            f"{min(timings):.0f}",
            f"{med:.0f}",
            f"{mean(timings):.0f}",
            f"{stdev(timings):.0f}" if len(timings) > 1 else "—",
            f"{met} < {target_ms} ms",
        )

        results[model_tag] = {
            "mode": mode.value,
            "runs": timings,
            "min_ms": min(timings),
            "median_ms": med,
            "mean_ms": mean(timings),
            "stdev_ms": stdev(timings) if len(timings) > 1 else 0.0,
            "target_ms": target_ms,
            "target_met": med <= target_ms,
        }

    console.print()
    console.print(summary_table)

    # Hardware context
    results["hardware"] = {
        "device":        HARDWARE.device,
        "compute_type":  HARDWARE.compute_type,
        "gpu_name":      HARDWARE.gpu_name,
        "gpu_memory_gb": HARDWARE.gpu_memory_gb,
        "cpu_threads":   HARDWARE.num_cpu_threads,
    }

    out_path.write_text(json.dumps(results, indent=2))
    console.print(f"\n[dim]Results saved to {out_path}[/dim]")


if __name__ == "__main__":
    app()
