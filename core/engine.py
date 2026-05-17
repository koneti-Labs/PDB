"""
core/engine.py -- Phase 2+3: GemmaEngine fully implemented via Ollama

Competition rule: ONLY gemma4:* model tags are permitted.
All inference runs locally -- nothing leaves the device.

InferenceMode routing:
  FAST_TRANSLATION      -> gemma4:e2b  (fast, ~2B params)
  REASONING_EXTRACTION  -> gemma4:e4b  (extended reasoning, think=True capable)

Performance notes:
  * keep_alive holds the model in (V)RAM between requests, avoiding the
    20-60s cold-load penalty that was producing 1-3 minute response times.
  * num_predict caps output length so translations cannot run away into a
    long generation.
  * warmup() preloads both models at server start so the first user request
    is not a cold start.  The class-level _warmed_up flag ensures each model
    is only warmed once per process regardless of how many CLI sub-commands
    call warmup().
"""
from __future__ import annotations

from enum import Enum
from typing import Any

import ollama
from rich.console import Console

# Static imports — Ollama connection params used once at construction time;
# safe to capture at import since they never change at runtime.
from config.settings import (
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    OLLAMA_TIMEOUT_REASONING,
)

# Dynamic import — generation knobs are read from this module at CALL TIME
# (not captured at import time) so that runtime patches applied by the
# notebook patch cell or by server.py self-healing take immediate effect.
# Previously the "from … import" approach captured GEMMA_NUM_CTX_FAST=512
# at import time, making runtime patches invisible to generate().
import config.settings as _cfg

console = Console()


class InferenceMode(str, Enum):
    FAST_TRANSLATION = "fast_translation"
    REASONING_EXTRACTION = "reasoning_extraction"


# Kaggle Gemma 4 Impact Challenge -- model mapping (must not change)
# Per project rules in CLAUDE.md:
#   FAST_TRANSLATION     -> gemma4:e2b
#   REASONING_EXTRACTION -> gemma4:e4b
MODELS: dict[InferenceMode, str] = {
    InferenceMode.FAST_TRANSLATION: "gemma4:e2b",
    InferenceMode.REASONING_EXTRACTION: "gemma4:e4b",
}

# Labels appended to prompts that Gemma 4 sometimes echoes back.
# _clean_response() strips these from the beginning of the output.
_PROMPT_LABELS: tuple[str, ...] = (
    "English for doctor:",
    "Hindi for patient:",
    "Telugu for patient:",
    "Kannada for patient:",
    "Tamil for patient:",
    # Prescription summary prompts end with "{language} explanation for patient:"
    "explanation for patient:",
    "for patient:",
    "Translation:",
    "Answer:",
)

import re as _re  # noqa: E402  (module-level, used by _clean_response only)


def _strip_think_tags(text: str) -> str:
    """
    Remove Gemma 4 extended-thinking blocks from raw model output.

    When think=True is active (triage mode) or when gemma4:e4b spontaneously
    emits reasoning traces, the output contains <think>...</think> blocks.
    These must be stripped before any downstream processing:
      - Translation callers would expose raw XML-like reasoning to patients.
      - JSON parsers in triage/prescription would match { } inside the block.

    Handles both complete blocks and truncated blocks (no closing tag due to
    num_predict cap).
    """
    # Remove complete blocks (non-greedy so multiple blocks are all removed)
    cleaned = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL | _re.IGNORECASE)
    # Remove unclosed block (truncated at token limit)
    cleaned = _re.sub(r"<think>.*$", "", cleaned, flags=_re.DOTALL | _re.IGNORECASE)
    return cleaned.strip()


def _clean_response(text: str) -> str:
    """
    Clean the raw Gemma 4 response before returning to callers.

    Steps (in order):
    1. Strip <think>...</think> extended-reasoning blocks — these must never
       be shown to patients or passed to JSON parsers.
    2. Strip echoed prompt labels from the very start of the output.
       Gemma 4 sometimes repeats the label that ends the prompt (e.g.
       "Hindi for patient:") before producing the actual translation.

    Both steps are applied unconditionally so callers never see internal
    model artefacts.
    """
    # Step 1: strip think-mode reasoning traces
    stripped = _strip_think_tags(text)

    # Step 2: strip echoed prompt labels (case-insensitive, prefix match only)
    for label in _PROMPT_LABELS:
        if stripped.lower().startswith(label.lower()):
            stripped = stripped[len(label):].strip()
            break
    return stripped


class GemmaEngine:
    """Local inference via Ollama using Gemma 4 exclusively."""

    # Class-level warmup registry: tracks which models have already been
    # loaded into Ollama (V)RAM in this process.  Keyed by model tag string.
    # This means pdb bridge, pdb triage, pdb reassure etc. all share the
    # same warmup state — the second sub-command to call warmup() is a no-op.
    _warmed_up: dict[str, bool] = {}

    def __init__(self) -> None:
        self._client = ollama.Client(
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT,
        )
        # Separate client for slow REASONING_EXTRACTION calls (think=True)
        self._reasoning_client = ollama.Client(
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_REASONING,
        )

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        mode: InferenceMode = InferenceMode.FAST_TRANSLATION,
        temperature: float | None = None,
        num_ctx: int | None = None,
        num_predict: int | None = None,
        think: bool = False,
        use_reasoning_client: bool = False,
    ) -> str:
        """
        Generate text using the Gemma 4 model mapped to *mode*.

        Parameters
        ----------
        prompt : str
            Full prompt string.
        mode : InferenceMode
            FAST_TRANSLATION -> gemma4:e2b, REASONING_EXTRACTION -> gemma4:e4b.
        temperature : float or None
            Override GEMMA_TEMPERATURE.
        num_ctx : int or None
            Override GEMMA_NUM_CTX / GEMMA_NUM_CTX_FAST (mode-dependent default).
        num_predict : int or None
            Override the output token cap (GEMMA_NUM_PREDICT_FAST or
            GEMMA_NUM_PREDICT_REASONING).  Use this when the default cap is
            too small for a particular task (e.g. prescription summary
            translation with many medicines needs more than 512 tokens).
        think : bool
            Enable extended reasoning (gemma4:e4b thinking mode).

        Returns
        -------
        str
            Stripped, label-cleaned response text, or raises RuntimeError on failure.
        """
        model = MODELS[mode]
        # Use the long-timeout client for reasoning calls
        client = (
            self._reasoning_client
            if mode == InferenceMode.REASONING_EXTRACTION
            else self._client
        )

        # Mode-specific defaults: fast translation uses a smaller context
        # window and tighter output cap; reasoning gets the full window.
        # Read from _cfg at call time so runtime patches are immediately effective.
        is_fast = mode == InferenceMode.FAST_TRANSLATION
        default_ctx = _cfg.GEMMA_NUM_CTX_FAST if is_fast else _cfg.GEMMA_NUM_CTX
        default_predict = (
            _cfg.GEMMA_NUM_PREDICT_FAST if is_fast else _cfg.GEMMA_NUM_PREDICT_REASONING
        )

        options: dict[str, Any] = {
            "temperature": temperature if temperature is not None else _cfg.GEMMA_TEMPERATURE,
            "top_p": _cfg.GEMMA_TOP_P,
            "num_ctx": num_ctx if num_ctx is not None else default_ctx,
            # Cap output length so the model cannot run off generating
            # paragraphs of trailing text.  Tuned for short clinical replies.
            # Callers may override via num_predict (e.g. prescription summary
            # translation needs more tokens than the fast-mode default of 512).
            "num_predict": num_predict if num_predict is not None else default_predict,
        }
        generate_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "options": options,
            # Hold the model in (V)RAM between calls so we do not pay the
            # cold-load tax on every request.  This is the single biggest
            # win for end-to-end latency.
            "keep_alive": _cfg.GEMMA_KEEP_ALIVE,
        }
        if think:
            generate_kwargs["think"] = True

        try:
            response = client.generate(**generate_kwargs)
        except TypeError as exc:
            # Older Ollama SDK (<0.5) does not support the 'think' keyword.
            # Retry without it so the call still succeeds.
            if "think" in str(exc) and "think" in generate_kwargs:
                generate_kwargs.pop("think")
                try:
                    response = client.generate(**generate_kwargs)
                except Exception as exc2:
                    error_msg = str(exc2)
                    if "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                        raise RuntimeError(
                            f"Model not found: {model}\n"
                            f"Available models can be checked with: ollama list\n"
                            f"Pull the model with: ollama pull {model}\n"
                            f"Original error: {exc2}"
                        ) from exc2
                    else:
                        raise RuntimeError(
                            f"Ollama inference failed (model={model}): {exc2}\n"
                            "Make sure Ollama is running and the model is pulled:\n"
                            f"  ollama pull {model}"
                        ) from exc2
            else:
                raise RuntimeError(
                    f"Ollama inference failed (model={model}): {exc}\n"
                    "Make sure Ollama is running and the model is pulled:\n"
                    f"  ollama pull {model}"
                ) from exc
        except Exception as exc:
            error_msg = str(exc)
            if "connection" in error_msg.lower() or "refused" in error_msg.lower():
                raise RuntimeError(
                    f"Cannot connect to Ollama at {OLLAMA_HOST}\n"
                    "Make sure Ollama is running:\n"
                    f"  ollama serve"
                ) from exc
            elif "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                raise RuntimeError(
                    f"Model not found: {model}\n"
                    f"Available models can be checked with: ollama list\n"
                    f"Pull the model with: ollama pull {model}"
                ) from exc
            else:
                raise RuntimeError(
                    f"Ollama inference failed (model={model}): {exc}\n"
                    "Make sure Ollama is running and the model is pulled:\n"
                    f"  ollama pull {model}"
                ) from exc

        # response.response is a str (Ollama SDK >= 0.2 returns a Pydantic
        # SubscriptableBaseModel, so both response["response"] and
        # response.response work; prefer attribute access for clarity).
        raw_text = response.response if hasattr(response, "response") else response["response"]
        return _clean_response(raw_text)

    # ------------------------------------------------------------------
    # Phase 2: Translation helpers
    # ------------------------------------------------------------------

    def translate_patient_to_doctor(self, text: str, source_lang: str) -> str:
        """
        Translate patient speech to medical English for the doctor.

        Parameters
        ----------
        source_lang : str
            ISO 639-1 code OR display name (both accepted).  ISO codes are
            mapped to display names automatically so the prompt reads
            "from Hindi" rather than "from hi".
        """
        from config.languages import LANGUAGE_DISPLAY
        from translation.prompts import PATIENT_TO_DOCTOR_PROMPT

        lang_name = LANGUAGE_DISPLAY.get(source_lang, source_lang)
        prompt = PATIENT_TO_DOCTOR_PROMPT.format(language=lang_name, text=text)
        return self.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)

    def translate_doctor_to_patient(self, text: str, target_lang: str) -> str:
        """
        Translate doctor's English response back to patient's language.

        Parameters
        ----------
        target_lang : str
            ISO 639-1 code OR display name.  ISO codes are mapped to display
            names so the prompt reads "into Hindi" rather than "into hi".
        """
        from config.languages import LANGUAGE_DISPLAY
        from translation.prompts import DOCTOR_TO_PATIENT_PROMPT

        lang_name = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = DOCTOR_TO_PATIENT_PROMPT.format(language=lang_name, text=text)
        return self.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)

    # ------------------------------------------------------------------
    # Phase 3: Emergency triage extraction
    # ------------------------------------------------------------------

    def emergency_triage(self, text: str, lang_code: str = "en") -> str:
        """
        Extract structured triage JSON from patient speech.

        Uses REASONING_EXTRACTION (gemma4:e4b) with think=True.
        Returns the raw JSON string; caller parses it
        (see translation.triage.TriageService).

        Parameters
        ----------
        text : str
            Raw Whisper transcript in the patient's language.
        lang_code : str
            ISO 639-1 code (hi/te/kn/en/ta).

        Returns
        -------
        str
            Raw JSON string from Gemma 4.
        """
        from config.languages import LANGUAGE_DISPLAY
        from translation.prompts import EMERGENCY_TRIAGE_PROMPT

        lang_name = LANGUAGE_DISPLAY.get(lang_code, lang_code)
        prompt = EMERGENCY_TRIAGE_PROMPT.format(language=lang_name, text=text)
        return self.generate(
            prompt,
            mode=InferenceMode.REASONING_EXTRACTION,
            think=_cfg.TRIAGE_THINK_MODE,
        )

    # ------------------------------------------------------------------
    # Phase 4: Prescription OCR
    # ------------------------------------------------------------------

    def transcribe_prescription(self, image_path: str) -> str:
        """
        OCR a prescription image via Gemma 4 multimodal vision.

        Uses REASONING_EXTRACTION (gemma4:e4b) with the image passed as a
        base64-encoded attachment.  Returns raw JSON string; caller parses it.

        Parameters
        ----------
        image_path : str
            Path to the prescription image (JPEG/PNG).

        Returns
        -------
        str
            Raw JSON string from Gemma 4.
        """
        import base64
        from translation.prompts import PRESCRIPTION_OCR_PROMPT

        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        model = MODELS[InferenceMode.REASONING_EXTRACTION]
        options: dict[str, Any] = {
            "temperature": 0.1,
            "top_p": _cfg.GEMMA_TOP_P,
            "num_ctx": _cfg.GEMMA_NUM_CTX,
            "num_predict": _cfg.GEMMA_NUM_PREDICT_REASONING,
        }
        try:
            response = self._reasoning_client.generate(
                model=model,
                prompt=PRESCRIPTION_OCR_PROMPT,
                images=[image_b64],
                options=options,
                keep_alive=_cfg.GEMMA_KEEP_ALIVE,
            )
            raw_text = response.response if hasattr(response, "response") else response["response"]
            # Strip any <think>...</think> blocks the vision model may emit
            # before the JSON output — these break downstream JSON parsing.
            return _strip_think_tags(raw_text.strip())
        except Exception as exc:
            raise RuntimeError(
                f"Prescription OCR failed (model={model}): {exc}\n"
                f"Make sure Ollama is running and {model} is pulled."
            ) from exc

    # ------------------------------------------------------------------
    # Phase 4b: Prescription summary translation
    # ------------------------------------------------------------------

    def translate_prescription_summary(
        self, prescription_text: str, target_lang: str
    ) -> str:
        """
        Translate a plain-English prescription summary into the patient's language.

        Uses FAST_TRANSLATION (gemma4:e2b).  Called by
        PrescriptionService.translate_summary() after OCR extraction.

        Parameters
        ----------
        prescription_text : str
            Human-readable English summary of extracted medicines + metadata.
        target_lang : str
            ISO 639-1 code (hi/te/kn/ta).  "en" is a no-op (caller checks first).

        Returns
        -------
        str
            Patient-friendly explanation in the target language.
        """
        from config.languages import LANGUAGE_DISPLAY
        from translation.prompts import PRESCRIPTION_SUMMARY_PROMPT

        lang_name = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = PRESCRIPTION_SUMMARY_PROMPT.format(
            language=lang_name,
            prescription_text=prescription_text,
        )
        # Use a higher token budget than the fast-translation default (512).
        # A prescription with 3-5 medicines described in an Indic language
        # (which is naturally wordier than English) can easily exceed 512
        # tokens.  1024 gives comfortable headroom while keeping latency low.
        return self.generate(
            prompt,
            mode=InferenceMode.FAST_TRANSLATION,
            num_predict=1024,
            use_reasoning_client=True,
        )

    # ------------------------------------------------------------------
    # Phase 5: Emergency reassurance
    # ------------------------------------------------------------------

    def emergency_reassurance(self, phrase: str, target_lang: str) -> str:
        """
        Translate an English emergency reassurance phrase into the patient's language.

        Uses FAST_TRANSLATION (gemma4:e2b) for speed -- reassurance must arrive quickly.

        Parameters
        ----------
        phrase : str
            Short English reassurance message (e.g. "Help is coming, stay calm.").
        target_lang : str
            ISO 639-1 code of the patient's language (hi/te/kn/en/ta).

        Returns
        -------
        str
            Translated reassurance in the patient's language.
        """
        from config.languages import LANGUAGE_DISPLAY
        from translation.prompts import EMERGENCY_REASSURANCE_PROMPT

        lang_name = LANGUAGE_DISPLAY.get(target_lang, target_lang)
        prompt = EMERGENCY_REASSURANCE_PROMPT.format(language=lang_name, phrase=phrase)
        return self.generate(prompt, mode=InferenceMode.FAST_TRANSLATION)

    # ------------------------------------------------------------------
    # Warmup -- preload models so the first user request is not a cold start
    # ------------------------------------------------------------------

    def warmup(self) -> dict[str, bool]:
        """
        Send a 1-token generate to every configured model so Ollama loads
        them into (V)RAM up-front.  keep_alive then pins them there.

        Deduplication: models already warmed in this Python process are
        skipped.  This means it is safe to call warmup() at the top of every
        CLI sub-command — only the very first call pays any cost; subsequent
        calls return immediately with the cached results.

        Errors are swallowed per-model so a missing model does not break
        server startup -- it will surface again at the first real call with
        a clearer error message.

        Returns
        -------
        dict[str, bool]
            Map of model tag -> warmup success (True if already warmed or just
            warmed successfully; False if Ollama could not load the model).
        """
        results: dict[str, bool] = {}
        any_new = False

        for mode, model in MODELS.items():
            # Skip models already warmed in this process
            if GemmaEngine._warmed_up.get(model):
                results[model] = True
                continue

            any_new = True
            client = (
                self._reasoning_client
                if mode == InferenceMode.REASONING_EXTRACTION
                else self._client
            )
            try:
                client.generate(
                    model=model,
                    prompt="ok",
                    options={"num_predict": 1, "temperature": 0.0},
                    keep_alive=_cfg.GEMMA_KEEP_ALIVE,
                )
                console.print(f"[green]  ✓ Warmed up {model}[/green]")
                GemmaEngine._warmed_up[model] = True
                results[model] = True
            except Exception as exc:
                console.print(f"[yellow]  ⚠ Warmup skipped for {model}: {exc}[/yellow]")
                GemmaEngine._warmed_up[model] = False
                results[model] = False

        if not any_new:
            console.print("[dim]  (Gemma 4 already loaded — skipping warmup)[/dim]")

        return results

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def check_connectivity(self) -> dict[str, bool]:
        """
        Verify Ollama is reachable and both Gemma 4 models are available.

        Uses .models attribute on the Pydantic ListResponse object returned
        by ollama.Client.list() (SDK >= 0.2).
        """
        results: dict[str, bool] = {}
        try:
            models_resp = self._client.list()
            # models_resp is a ListResponse (SubscriptableBaseModel).
            # Access via .models attribute to avoid relying on dict .get().
            model_list = (
                models_resp.models
                if hasattr(models_resp, "models")
                else models_resp.get("models", [])
            )
            available: set[str] = set()
            for m in model_list:
                # Each entry is a ListResponse.Model with a .model field (Optional[str])
                tag = m.model if hasattr(m, "model") else m.get("model")
                if tag:
                    available.add(tag)
        except Exception:
            return {model: False for model in MODELS.values()}

        return {model: model in available for model in MODELS.values()}
