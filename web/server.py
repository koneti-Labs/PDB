"""
web/server.py

Flask web server for PatientDoctorBridge Phase 6 UI.

Endpoints
---------
GET  /                          -> index.html (single-page app)
POST /api/bridge/patient        -> patient speech -> doctor English
POST /api/bridge/doctor         -> doctor English -> patient language
POST /api/triage                -> patient speech -> SSE stream of triage events
POST /api/reassure              -> English phrase -> patient language
POST /api/prescription          -> prescription image -> medicines JSON
POST /api/tts                   -> text + language -> WAV audio (MMS-TTS, no OS voice pack needed)

All audio endpoints accept multipart/form-data with:
  - audio: binary blob (WebM/OGG from MediaRecorder or WAV)
  - language: ISO 639-1 language code (hi/te/kn/ta/en)

Concurrency model
-----------------
  • Flask runs with threaded=True — each HTTP request gets its own OS thread.
  • All heavy operations (CTranslate2 Whisper, Ollama HTTP, PyTorch TTS) release
    the GIL, so multiple threads run in true parallel on multi-core hardware.
  • A bounded ThreadPoolExecutor (_inference_pool) caps concurrent inference
    threads at INFERENCE_POOL_SIZE to prevent resource exhaustion under load.
  • /api/triage uses Server-Sent Events (SSE) so the transcript appears in the
    UI ~2-4 s after recording, rather than after the full 15-30 s triage cycle.
  • TTS models are optionally preloaded in parallel at startup via TTSService.prewarm()
    (configured by TTS_PREWARM_LANGS in config/settings.py).

Privacy: temp files deleted immediately after processing.
"""
from __future__ import annotations

import json
import os
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file, stream_with_context
from rich.console import Console

from audio.handler import AudioHandler
from audio.tts import TTSService
from config.settings import INFERENCE_POOL_SIZE, WEB_HOST, WEB_PORT
from core.engine import GemmaEngine
from translation.prescription import PrescriptionService
from translation.reassurance import REASSURANCE_PHRASES, ReassuranceService
from translation.service import TranslationService
from translation.triage import TriageService

console = Console()

# ── Self-healing settings patch ───────────────────────────────────────────────
# Guarantee the correct context window is in effect for gemma4:e2b regardless
# of when config.settings was first imported by this Python process.
import config.settings as _cfg_patch  # noqa: E402

_cfg_patch.GEMMA_NUM_CTX_FAST     = 2048
_cfg_patch.GEMMA_NUM_PREDICT_FAST = 512

# ── Bounded inference thread pool ─────────────────────────────────────────────
# Flask's threaded=True spawns one OS thread per request.  Under concurrent
# load (e.g. 10 users at once) that creates 10 threads all hammering Ollama
# and Whisper simultaneously.  A bounded pool caps the number of CONCURRENT
# inference calls; excess requests queue instead of thrashing the system.
#
# Why ThreadPoolExecutor and not asyncio?
#   • Whisper (CTranslate2), Ollama HTTP (requests), and TTS (PyTorch) all
#     release the GIL during their heavy work → OS threads achieve TRUE
#     parallelism on multi-core hardware without needing the async ecosystem.
#   • asyncio would require rewriting every blocking call with httpx/aiofiles.
#     ThreadPoolExecutor gives ~90% of the concurrency benefit with zero
#     changes to the inference layer.
#
# Future migration path to full async (FastAPI + ollama.AsyncClient):
#   1. Replace Flask with FastAPI.
#   2. Replace ollama.Client with ollama.AsyncClient in core/engine.py.
#   3. Wrap Whisper and TTS calls in asyncio.get_event_loop().run_in_executor()
#      with a dedicated ThreadPoolExecutor for CPU-bound work.
#   This would allow a single-thread event loop to handle N concurrent HTTP
#   connections, reducing per-process memory from O(N threads) to O(1 thread).
_inference_pool = ThreadPoolExecutor(
    max_workers=INFERENCE_POOL_SIZE,
    thread_name_prefix="pdb-infer",
)

# Lazy-initialised singletons.  Building these once at server startup (instead
# of per-request) shaves ~5-50ms off every request and eliminates the
# per-request "Loading Whisper..." line we used to see in the logs.
_engine: GemmaEngine | None = None
_audio_handler: AudioHandler | None = None
_translation_svc: TranslationService | None = None
_triage_svc: TriageService | None = None
_reassure_svc: ReassuranceService | None = None
_prescription_svc: PrescriptionService | None = None
_tts_svc: TTSService | None = None


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


def _get_tts_svc() -> TTSService:
    global _tts_svc
    if _tts_svc is None:
        _tts_svc = TTSService()
    return _tts_svc


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

    # ------------------------------------------------------------------ triage (SSE)
    @app.route("/api/triage", methods=["POST"])
    def triage():
        """
        Transcribe patient audio and extract triage JSON.

        Returns a Server-Sent Events (SSE) stream so the browser receives
        incremental progress rather than waiting 15-35 s for a single JSON blob.

        SSE event sequence
        ------------------
        1. {"status": "transcribing"}
              → sent immediately so the UI can show a spinner with a label.
        2. {"status": "transcribed", "transcript": "...", "language": "hi"}
              → fired after Whisper (~2-4 s).  The UI renders the transcript
                immediately — the user can read what was heard while Gemma 4
                is still thinking.
        3. {"status": "analysing", "message": "Gemma 4 extracting triage..."}
              → short status update while REASONING_EXTRACTION (think=True) runs.
        4a. {"status": "complete", "transcript": "...", "language": "hi",
              "triage": {...}}
              → sent after Gemma 4 returns (~10-30 s total).
        4b. {"status": "error", "error": "..."}
              → sent on any exception; the UI shows an error message.

        Why SSE and not WebSocket?
          SSE is unidirectional (server → client), works over plain HTTP/1.1,
          needs no handshake upgrade, and is natively supported by the browser
          EventSource API.  For a streaming progress use case (server pushes
          intermediate results; client never sends back during processing) SSE
          is simpler and more appropriate than WebSocket.

        Why not async/await?
          The heavy work (CTranslate2 Whisper + Ollama HTTP) releases the GIL,
          so OS threads achieve true parallelism without asyncio.  Flask's
          stream_with_context() drives the SSE generator from the request
          thread; _inference_pool.submit() off-loads each blocking step to a
          bounded worker thread so the Flask thread is free to flush SSE
          events while inference runs.
        """
        audio_file = request.files.get("audio")
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        tmp_path = _save_temp_audio(audio_file)
        if tmp_path is None:
            return jsonify({"error": "Failed to save audio"}), 500

        def _sse(payload: dict) -> str:
            """Format a dict as a single SSE data line."""
            return "data: " + json.dumps(payload) + "\n\n"

        # Check if user locked a language (skip auto-detection)
        locked_lang = request.form.get("language", "").strip().lower()

        def generate():
            try:
                # ── Step 1: Transcription ─────────────────────────────────
                yield _sse({"status": "transcribing"})

                # Submit to the bounded inference pool so this blocking call
                # does not pin the Flask thread indefinitely.
                handler = _get_audio_handler()
                if locked_lang and locked_lang in {"hi", "te", "kn", "ta", "en"}:
                    transcribe_future = _inference_pool.submit(
                        handler.transcribe_locked, tmp_path, locked_lang
                    )
                else:
                    transcribe_future = _inference_pool.submit(
                        handler.transcribe, tmp_path
                    )
                result = transcribe_future.result()   # blocks until Whisper finishes
                transcript   = result["text"]
                detected_lang = result["language"]

                console.print(
                    f"[dim]Triage: transcribed ({detected_lang}): "
                    f"{transcript[:60]}{'…' if len(transcript) > 60 else ''}[/dim]"
                )
                yield _sse({
                    "status":     "transcribed",
                    "transcript": transcript,
                    "language":   detected_lang,
                })

                # ── Step 2: Gemma 4 triage extraction ────────────────────
                yield _sse({
                    "status":  "analysing",
                    "message": "Gemma 4 extracting triage data (think=True)…",
                })

                svc = _get_triage_svc()
                triage_future = _inference_pool.submit(
                    svc.extract, transcript, detected_lang
                )
                triage_result = triage_future.result()  # blocks until Gemma returns

                console.print("[dim]Triage: extraction complete.[/dim]")
                yield _sse({
                    "status":     "complete",
                    "transcript": transcript,
                    "language":   detected_lang,
                    "triage":     dict(triage_result),
                })

            except Exception as exc:
                traceback.print_exc()
                yield _sse({"status": "error", "error": str(exc)})
            finally:
                # Always clean up the temp audio file
                _delete_temp(tmp_path)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                # Prevent proxies and nginx from buffering the SSE stream.
                "Cache-Control":   "no-cache",
                "X-Accel-Buffering": "no",
                # Allow cross-origin clients (dev tools, Kaggle notebook)
                "Access-Control-Allow-Origin": "*",
            },
        )

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
        """OCR a prescription image and optionally translate the result.

        Form fields
        -----------
        image    : binary — prescription photo (JPEG / PNG / WebP)
        language : str    — ISO 639-1 target language for patient summary
                           (hi/te/kn/ta/en).  Defaults to "en" (English only).
        """
        image_file = request.files.get("image")
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400

        # Target language for the patient-friendly summary translation
        lang_code = (request.form.get("language") or "en").strip().lower()

        suffix = Path(image_file.filename or "rx.jpg").suffix or ".jpg"
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(tmp_fd)
            image_file.save(tmp_path)
            svc = _get_prescription_svc()
            result = svc.extract(tmp_path)
            response_data = dict(result)

            # Phase 4b: translate the prescription summary if a non-English
            # language was requested.  The translated text is patient-friendly
            # and uses simple everyday language.
            if lang_code and lang_code != "en":
                console.print(
                    f"[dim]Translating prescription summary to {lang_code}…[/dim]"
                )
                try:
                    translated = svc.translate_summary(result, lang_code)
                    response_data["translated_summary"] = translated
                    response_data["summary_language"] = lang_code
                except Exception as trans_exc:
                    # Never let a translation failure hide the OCR result
                    console.print(
                        f"[yellow]⚠ Prescription translation failed: {trans_exc}[/yellow]"
                    )
                    traceback.print_exc()
                    response_data["translated_summary"] = ""
                    response_data["summary_language"] = lang_code
                    response_data["translation_error"] = str(trans_exc)

            return jsonify(response_data)
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500
        finally:
            _delete_temp(tmp_path)

    # ------------------------------------------------------------------ tts
    @app.route("/api/tts", methods=["POST"])
    def tts():
        """
        Server-side TTS using Facebook MMS-TTS (no OS voice pack required).

        JSON body
        ---------
        {
          "text":     "<text to speak>",
          "language": "<ISO 639-1 code: hi | te | ta | kn | en>"
        }

        Returns audio/wav binary on success.
        First call per language downloads the ~80 MB model from HuggingFace
        and caches it; subsequent calls are fast (~1-2 s).
        """
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()
        lang_code = data.get("language", "hi").strip().lower()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        try:
            svc = _get_tts_svc()
            console.print(f"[dim]MMS-TTS: synthesizing {len(text)} chars in '{lang_code}'…[/dim]")
            wav_bytes, _sr = svc.synthesize(text, lang_code)
            import io as _io
            buf = _io.BytesIO(wav_bytes)
            buf.seek(0)
            return send_file(buf, mimetype="audio/wav", as_attachment=False)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except Exception as exc:
            traceback.print_exc()
            return jsonify({"error": str(exc)}), 500

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
        if role == "doctor":
            # Doctor's audio is always English — lock Whisper, skip detection.
            result = handler.transcribe_locked(tmp_path, language="en")
        else:
            # Patient turn: use auto-detection so the system automatically
            # recognises whichever supported language the patient speaks
            # (hi, te, kn, ta, en).  The transcribe() method has built-in
            # constrained fallback to these five languages.
            result = handler.transcribe(tmp_path)
        transcript = result["text"]
        detected_lang = result["language"]

        svc = _get_translation_svc()
        if role == "patient":
            console.print(f"[dim]Translating patient ({detected_lang}) → doctor (en)…[/dim]")
            translation = svc.patient_to_doctor(transcript, detected_lang)
            if not translation:
                console.print("[yellow]⚠ Warning: Empty translation returned[/yellow]")
            return jsonify({
                "transcript": transcript,
                "detected_language": detected_lang,
                "translation": translation,
                "direction": "patient_to_doctor",
            })
        else:
            console.print(f"[dim]Translating doctor (en) to patient ({lang_code})...[/dim]")
            translation = svc.doctor_to_patient(transcript, lang_code)
            if not translation:
                console.print("[yellow]⚠ Warning: Empty translation returned[/yellow]")
            return jsonify({
                "transcript": transcript,
                "detected_language": "en",
                "translation": translation,
                "direction": "doctor_to_patient",
                "target_language": lang_code,
            })
    except Exception as exc:
        console.print(f"[red]✗ Translation error ({role}): {exc}[/red]")
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
    console.print("[bold cyan]PatientDoctorBridge Web UI[/bold cyan]")
    console.print(f"[dim]Open in browser: http://{h}:{p}/[/dim]")
    console.print("[dim]All inference runs locally via Ollama (Gemma 4).[/dim]")

    # ── Parallel startup warmup ───────────────────────────────────────────────
    # Three independent warmup jobs run simultaneously using plain threads
    # (not the inference pool, so the pool stays free for the first request).
    #
    #   Thread A — Whisper: loads the ASR model into RAM.
    #   Thread B — Gemma 4: sends a 1-token ping to each model so Ollama
    #              loads them into (V)RAM.  keep_alive pins them there.
    #   Thread C — TTS (optional): preloads MMS-TTS language models in their
    #              own parallel sub-pool (controlled by TTS_PREWARM_LANGS).
    #
    # All three release the GIL during their heavy work (disk I/O, HTTP,
    # PyTorch), so they run in true parallel on multi-core hardware.
    import threading

    import config.settings as _cfg_startup

    console.print(
        f"[dim]Starting parallel warmup: Whisper + Gemma 4"
        f"{' + TTS' if _cfg_startup.TTS_PREWARM_LANGS else ''}…[/dim]"
    )

    def _gemma_warmup() -> None:
        try:
            _get_engine().warmup()
        except Exception as exc:
            console.print(f"[yellow]Gemma warmup skipped: {exc}[/yellow]")

    def _tts_prewarm() -> None:
        """Parallel-load TTS models for all configured languages."""
        langs = _cfg_startup.TTS_PREWARM_LANGS
        if not langs:
            return
        console.print(f"[dim]TTS prewarm: loading {langs} in parallel…[/dim]")
        try:
            results = TTSService.prewarm(langs)
            for lang, ok in results.items():
                icon = "✓" if ok else "⚠"
                color = "green" if ok else "yellow"
                console.print(f"[{color}]  {icon} TTS {lang}[/{color}]")
        except Exception as exc:
            console.print(f"[yellow]TTS prewarm failed: {exc}[/yellow]")

    whisper_thread = threading.Thread(target=_preload_audio_handler, daemon=True)
    gemma_thread   = threading.Thread(target=_gemma_warmup,          daemon=True)
    tts_thread     = threading.Thread(target=_tts_prewarm,            daemon=True)

    whisper_thread.start()
    gemma_thread.start()
    tts_thread.start()

    whisper_thread.join()
    gemma_thread.join()
    tts_thread.join()

    console.print(
        f"[green]Ready.[/green]  "
        f"[dim]Inference pool: {INFERENCE_POOL_SIZE} worker(s).[/dim]\n"
    )

    # use_reloader=False prevents the warmup from running twice in dev mode.
    app.run(host=h, port=p, debug=debug, use_reloader=False, threaded=True)


if __name__ == "__main__":
    run_server()
