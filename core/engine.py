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

from config.settings import (
    GEMMA_KEEP_ALIVE,
    GEMMA_NUM_CTX,
    GEMMA_NUM_CTX_FAST,
    GEMMA_NUM_PREDICT_FAST,
    GEMMA_NUM_PREDICT_REASONING,
    GEMMA_TEMPERATURE,
    GEMMA_TOP_P,
    OLLAMA_HOST,
    OLLAMA_TIMEOUT,
    OLLAMA_TIMEOUT_REASONING,
    TRIAGE_THINK_MODE,
)

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
    "for patient:",
    "Translation:",
    "Answer:",
)


def _clean_response(text: str) -> str:
    """
    Strip echoed prompt labels that Gemma 4 occasionally repeats at the
    very start of its output.

    For example, if the prompt ends with "English for doctor:" and the model
    echos that label before producing the actual translation, this function
    removes the label so the caller only sees the translation.
    """
    stripped = text.strip()
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
        think: bool = False,
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
            Override GEMMA_NUM_CTX.
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
        is_fast = mode == InferenceMode.FAST_TRANSLATION
        default_ctx = GEMMA_NUM_CTX_FAST if is_fast else GEMMA_NUM_CTX
        default_predict = (
            GEMMA_NUM_PREDICT_FAST if is_fast else GEMMA_NUM_PREDICT_REASONING
        )

        options: dict[str, Any] = {
            "temperature": temperature if temperature is not None else GEMMA_TEMPERATURE,
            "top_p": GEMMA_TOP_P,
            "num_ctx": num_ctx if num_ctx is not None else default_ctx,
            # Cap output length so the model cannot run off generating
            # paragraphs of trailing text.  Tuned for short clinical replies.
            "num_predict": default_predict,
        }
        generate_kwargs: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "options": options,
            # Hold the model in (V)RAM between calls so we do not pay the
            # cold-load tax on every request.  This is the single biggest
            # win for end-to-end latency.
            "keep_alive": GEMMA_KEEP_ALIVE,
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
            think=TRIAGE_THINK_MODE,
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
            "top_p": GEMMA_TOP_P,
            "num_ctx": GEMMA_NUM_CTX,
            "num_predict": GEMMA_NUM_PREDICT_REASONING,
        }
        try:
            response = self._reasoning_client.generate(
                model=model,
                prompt=PRESCRIPTION_OCR_PROMPT,
                images=[image_b64],
                options=options,
                keep_alive=GEMMA_KEEP_ALIVE,
            )
            raw_text = response.response if hasattr(response, "response") else response["response"]
            return raw_text.strip()
        except Exception as exc:
            raise RuntimeError(
                f"Prescription OCR failed (model={model}): {exc}\n"
                f"Make sure Ollama is running and {model} is pulled."
            ) from exc

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
                    keep_alive=GEMMA_KEEP_ALIVE,
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
        except Exception as exc:
            raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_HOST}: {exc}") from exc

        return {model: model in available for model in MODELS.values()}
