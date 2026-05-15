from .inference_engine import InferenceEngine
from .config import InferenceMode

class TranslationService:
    def __init__(self):
        self.engine = InferenceEngine()

    def patient_to_doctor(self, transcription: str, detected_lang: str) -> str:
        """Translate patient speech to clear medical English for doctor"""
        
        system_prompt = f"""You are a professional medical translator.
Translate the following patient's message from {detected_lang} to clear, accurate, and professional English.
Preserve medical symptoms, severity, and details accurately.
Do not add any extra information.

Patient ({detected_lang}):"""

        full_prompt = f"{system_prompt}\n\n{transcription}\n\nDoctor's Version:"

        translation = self.engine.generate(
            prompt=full_prompt,
            mode=InferenceMode.FAST_TRANSLATION,
            temperature=0.2
        )
        
        return translation