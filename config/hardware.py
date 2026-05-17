"""
config/hardware.py

Auto-detect available compute hardware (CPU / CUDA GPU) and return
optimal settings for Whisper (faster-whisper / CTranslate2) and Ollama.

Priority order:
  1. CUDA GPU  — float16, fastest inference
  2. CPU       — int8 quantized, universal fallback

TPU note: Kaggle TPU v3-8 runs Google's JAX stack and is NOT compatible
with CTranslate2 (Whisper) or Ollama.  If a TPU is present but CUDA is
absent, the code falls back gracefully to CPU for all inference.

Kaggle GPU reference:
  NVIDIA T4  — 16 GB VRAM  (most common)
  NVIDIA P100 — 16 GB VRAM
  NVIDIA A100 — 40 GB VRAM
  Raspberry Pi 5 + Hailo-10H NPU — CPU-only for Whisper/Ollama (Phase 7)

Usage
-----
    from config.hardware import HARDWARE, print_hardware_report
    print_hardware_report()

    # Fields:
    #   HARDWARE.device          -> "cuda" | "cpu"
    #   HARDWARE.compute_type    -> "float16" | "int8"
    #   HARDWARE.has_gpu         -> bool
    #   HARDWARE.gpu_name        -> str
    #   HARDWARE.gpu_memory_gb   -> float
    #   HARDWARE.num_cpu_threads -> int
    #   HARDWARE.ollama_gpu_layers -> int  (0 = CPU, 999 = all on GPU)
"""
from __future__ import annotations

import os
from typing import NamedTuple

from rich.console import Console
from rich.table import Table

console = Console()


class HardwareProfile(NamedTuple):
    """Immutable snapshot of detected hardware + recommended settings."""

    device: str            # "cuda" | "cpu"
    compute_type: str      # "float16" | "int8"
    has_gpu: bool
    gpu_name: str          # "" if no GPU
    gpu_memory_gb: float   # 0.0 if no GPU
    num_cpu_threads: int   # recommended CTranslate2 CPU threads
    ollama_gpu_layers: int # 999 = full GPU offload, 0 = CPU-only


def detect() -> HardwareProfile:
    """
    Probe available hardware and return the best settings.

    Never raises — falls back to CPU on any detection error so server
    startup is never blocked by a missing CUDA library.
    """
    cpu_count = os.cpu_count() or 4
    # CTranslate2 throughput saturates around 8 threads on modern CPUs.
    cpu_threads = min(cpu_count, 8)

    # ── CUDA GPU ──────────────────────────────────────────────────────────
    try:
        import torch  # bundled with faster-whisper CUDA wheel

        if torch.cuda.is_available():
            idx = 0
            gpu_name = torch.cuda.get_device_name(idx)
            total_mem = torch.cuda.get_device_properties(idx).total_memory
            gpu_gb = round(total_mem / (1024 ** 3), 1)

            # float16 works on all Kaggle GPUs (T4 / P100 / A100, sm >= 60).
            # Ollama gets full GPU offload when VRAM >= 6 GB.
            gpu_layers = 999 if gpu_gb >= 6.0 else 0

            return HardwareProfile(
                device="cuda",
                compute_type="float16",
                has_gpu=True,
                gpu_name=gpu_name,
                gpu_memory_gb=gpu_gb,
                num_cpu_threads=cpu_threads,
                ollama_gpu_layers=gpu_layers,
            )
    except Exception:
        pass  # torch not installed or CUDA unavailable — fall through to CPU

    # ── CPU fallback ───────────────────────────────────────────────────────
    return HardwareProfile(
        device="cpu",
        compute_type="int8",
        has_gpu=False,
        gpu_name="",
        gpu_memory_gb=0.0,
        num_cpu_threads=cpu_threads,
        ollama_gpu_layers=0,
    )


# Module-level singleton — computed once at import time so every caller
# sees the same profile without re-running CUDA detection.
HARDWARE: HardwareProfile = detect()


def print_hardware_report() -> None:
    """Print a Rich-formatted hardware summary to the terminal."""
    table = Table(
        title="PatientDoctorBridge — Hardware Profile",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Property", style="dim", min_width=18)
    table.add_column("Value")

    table.add_row("Device",          HARDWARE.device.upper())
    table.add_row("Compute type",    HARDWARE.compute_type)

    if HARDWARE.has_gpu:
        table.add_row("GPU",             HARDWARE.gpu_name)
        table.add_row("GPU VRAM",        f"{HARDWARE.gpu_memory_gb} GB")
        table.add_row("Ollama GPU layers", str(HARDWARE.ollama_gpu_layers))
    else:
        table.add_row("GPU",             "[yellow]Not available — CPU mode[/yellow]")

    table.add_row("CPU threads",     str(HARDWARE.num_cpu_threads))

    console.print(table)
    if HARDWARE.has_gpu:
        console.print(
            f"[green]✓ GPU detected: {HARDWARE.gpu_name} "
            f"({HARDWARE.gpu_memory_gb} GB). Whisper will use float16.[/green]"
        )
    else:
        console.print(
            "[yellow]⚠ No GPU detected. Whisper will use int8 on CPU. "
            "Expect slower transcription (~3-8x vs GPU).[/yellow]"
        )
