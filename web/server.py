"""
web/server.py

Flask web server for PatientDoctorBridge Phase 6 UI.

Endpoints
---------
GET  /                          -> index.html (single-page app)
POST /api/bridge/patient        -> patient speech -> doctor English
POST /api/bridge/doctor         -> doctor English -> patient language
POST /api/triage                -> patient speech -> triage JSON
POST /api/reassure              -> English phrase -> patient language
POST /api/prescription          -> prescription image -> medicines JSON

All audio endpoints accept multipart/form-data with:
  - audio: binary blob (WebM/OGG from MediaRecorder or WAV)
  - language: ISO 639-1 language code (hi/te/kn/ta/en)

Privacy: temp files deleted immediately after processing.
"""
from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

from flask import Flask, jsonify, request
from rich.console import Console

from audio.handler import AudioHandler
from config.settings import WEB_HOST, WEB_PORT
from core.engine import GemmaEngine
from translation.prescription import PrescriptionService
from translation.reassurance import REASSURANCE_PHRASES, ReassuranceService
from translation.service import TranslationService
from translation.triage import TriageService

console = Console()

# Lazy-initialised singletons.  Building these once at server startup (instead
# of per-request) shaves ~5-50ms off every request and eliminates the
# per-request "Loading Whisper..." line we used to see in the logs.
_engine: GemmaEngine | None = None
_audio_handler: AudioHandler | None = None
_translation_svc: TranslationService | None = None
_triage_svc: TriageService | None = None
_reassure_svc: ReassuranceService | None = None
_prescription_svc: PrescriptionService | None = None


def _get_engine() -> GemmaEngine:
    global _engine
    if _engine is None:
        _engine = GemmaEngine()
    return _engine


def _get_audio_handler() -> AudioHandler:
    global _audio_handler
    if _audio_handler is None:
        _audio_handler = AudioHandler()
    return _audio_handler


def _get_translation_svc() -> TranslationService:
    global _translation_svc
    if _translation_svc is None:
        _translation_svc = TranslationService(_get_engine())
    return _translation_svc


def _get_triage_svc() -> TriageService:
    global _triage_svc
    if _triage_svc is None:
        _triage_svc = TriageService(_get_engine())
    return _triage_svc


def _get_reassure_svc() -> ReassuranceService:
    global _reassure_svc
    if _reassure_svc is None:
        _reassure_svc = ReassuranceService(_get_engine())
    return _reassure_svc


def _get_prescription_svc() -> PrescriptionService:
    global _prescription_svc
    if _prescription_svc is None:
        _prescription_svc = PrescriptionService(_get_engine())
    return _prescription_svc


def _preload_audio_handler() -> None:
    """Force the Whisper model to load now (class-level singleton).

    Without this the first audio-bearing request pays ~5-15s loading
    Whisper.  Called once at server start.
    """
    try:
        AudioHandler._ensure_model()
        # Also build the handler singleton so the first request doesn't pay
        # any extra construction cost.
        _get_audio_handler()
    except Exception as exc:
        console.print(f"[yellow]Whisper preload skipped: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> Flask:
    static_dir = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")

    # ------------------------------------------------------------------ index
    @app.route("/")
    def index():
        index_path = static_dir / "index.html"
        with open(index_path, encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "text/html; charset=utf-8"}

    # ------------------------------------------------------------------ health
    @app.route("/api/health")
    def health():
        try:
            status = _get_engine().check_connectivity()
            return jsonify({"ok": True, "models": status})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 503

    # ------------------------------------------------------------------ phrase list
    @app.route("/api/phrases")
    def phrases():
        return jsonify([
            {"category": cat, "phrase": phrase}
            for cat, phrase in REASSURANCE_PHRASES
        ])

    # ------------------------------------------------------------------ bridge/patient
    @app.route("/api/bridge/patient", methods=["POST"])
    def bridge_patient():
        """Transcribe patient audio, then translate to doctor English."""
        return _handle_audio_translation("patient")

    # ------------------------------------------------------------------ bridge/doctor
    @app.route("/api/bridge/doctor", methods=["POST"])
    def bridge_doctor():
        """Transcribe doctor audio, then translate to patient language."""
        return _handle_audio_translation("doctor")

    # ------------------------------------------------------------------ triage
    @app.route("/api/triage", methods=["POST"])
    def triage():
        """Transcribe patient audio and extract triage JSON."""
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        tmp_path = _save_temp_audio(audio_file)
        if tmp_path is None:
            return jsonify({"error": "Failed to save audio"}), 500

        try:
            handler = _get_audio_handler()
            result = handler.transcribe(tmp_path)
            transcript = result["text"]
            detected_lang = result["language"]

            svc = _get_triage_svc()
            triage_result = svc.extract(transcript, detected_lang)

            return jsonify({
                "transcript": transcript,
                "language": detected_lang,
                "triage": dict(triage_result),
            })
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500
        finally:
            _delete_temp(tmp_path)

    # ------------------------------------------------------------------ reassure
    @app.route("/api/reassure", methods=["POST"])
    def reassure():
        """Translate an English phrase to patient language."""
        data = request.get_json(silent=True) or {}
        phrase = data.get("phrase", "").strip()
        lang_code = data.get("language", "hi")

        if not phrase:
            return jsonify({"error": "No phrase provided"}), 400

        try:
            svc = _get_reassure_svc()
            translated = svc.translate(phrase, lang_code)
            return jsonify({"original": phrase, "translated": translated, "language": lang_code})
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500

    # ------------------------------------------------------------------ prescription
    @app.route("/api/prescription", methods=["POST"])
    def prescription():
        """OCR a prescription image."""
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        suffix = Path(image_file.filename or "rx.jpg").suffix or ".jpg"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(tmp_fd)
            image_file.save(tmp_path)
            svc = _get_prescription_svc()
            result = svc.extract(tmp_path)
            return jsonify(dict(result))
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500
        finally:
            _delete_temp(tmp_path)

    return app


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _handle_audio_translation(role: str):
    """Shared handler for patient/doctor audio translation."""
    audio_file = request.files.get("audio")
    lang_code = request.form.get("language", "hi")

    if not audio_file:
        return jsonify({"error": "No audio file provided"}), 400

    tmp_path = _save_temp_audio(audio_file)
    if tmp_path is None:
        return jsonify({"error": "Failed to save audio"}), 500

    try:
        handler = _get_audio_handler()
        # Doctor's audio is always English -- skip Whisper language detection
        # entirely.  Saves ~1-2s per doctor turn on CPU.
        if role == "doctor":
            result = handler.transcribe_locked(tmp_path, language="en")
        else:
            result = handler.transcribe(tmp_path)
        transcript = result["text"]
        detected_lang = result["language"]

        svc = _get_translation_svc()
        if role == "patient":
            # TranslationService.patient_to_doctor() maps ISO code → display name
            # so the prompt reads "from Hindi" not "from hi".
            translation = svc.patient_to_doctor(transcript, detected_lang)
            return jsonify({
                "transcript": transcript,
                "detected_language": detected_lang,
                "translation": translation,
                "direction": "patient_to_doctor",
            })
        else:
            # For the doctor turn, lang_code is the patient's detected language
            # sent back by the frontend after the patient turn.
            translation = svc.doctor_to_patient(transcript, lang_code)
            return jsonify({
                "transcript": transcript,
                "detected_language": "en",
                "translation": translation,
                "direction": "doctor_to_patient",
                "target_language": lang_code,
            })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
    finally:
        _delete_temp(tmp_path)


def _save_temp_audio(audio_file) -> str | None:
    """Save uploaded audio to a temp WAV file; return path or None."""
    try:
        suffix = ".webm"
        fname = audio_file.filename or ""
        if fname.endswith(".wav"):
            suffix = ".wav"
        elif fname.endswith(".ogg"):
            suffix = ".ogg"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(tmp_fd)
        audio_file.save(tmp_path)
        return tmp_path
    except Exception:
        return None


def _delete_temp(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_server(
    host: str | None = None,
    port: int | None = None,
    debug: bool = False,
) -> None:
    h = host or WEB_HOST
    p = port or WEB_PORT
    app = create_app()
    console.print(f"[bold cyan]PatientDoctorBridge Web UI[/bold cyan]")
    console.print(f"[dim]Open in browser: http://{h}:{p}/[/dim]")
    console.print(f"[dim]All inference runs locally via Ollama (Gemma 4).[/dim]")

    # Warm up everything BEFORE serving requests so first-call latency is fast.
    # Run Whisper preload (CPU-bound, releases GIL in native code) and Gemma
    # warmup (HTTP I/O to Ollama) in parallel -- they don't contend for the
    # same resource, so wall-clock startup is roughly the slower of the two
    # instead of their sum.
    import threading
    console.print("[dim]Warming up Whisper + Gemma 4 in parallel (one-time)...[/dim]")

    def _gemma_warmup() -> None:
        try:
            _get_engine().warmup()
        except Exception as exc:
            console.print(f"[yellow]Gemma warmup skipped: {exc}[/yellow]")

    whisper_thread = threading.Thread(target=_preload_audio_handler, daemon=True)
    gemma_thread = threading.Thread(target=_gemma_warmup, daemon=True)
    whisper_thread.start()
    gemma_thread.start()
    whisper_thread.join()
    gemma_thread.join()
    console.print("[green]Ready.[/green]\n")

    # use_reloader=False prevents the warmup from running twice in dev mode.
    app.run(host=h, port=p, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    run_server()
