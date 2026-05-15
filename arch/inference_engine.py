import ollama
from rich.console import Console
from .config import MODELS, InferenceMode

console = Console()

class InferenceEngine:
    """Strictly enforces Gemma 4 usage as per Kaggle rules"""
    
    def __init__(self):
        self.current_model = None

    def generate(self, prompt: str, mode: InferenceMode = InferenceMode.FAST_TRANSLATION, temperature=0.3):
        """Generate response using only Gemma 4"""
        model = MODELS[mode]
        
        if model != self.current_model:
            console.print(f"[yellow]Loading {model}...[/yellow]")
            self.current_model = model

        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                }
            )
            return response['response'].strip()
        except Exception as e:
            console.print(f"[red]Ollama Error: {e}[/red]")
            console.print("[yellow]Make sure Ollama is running and Gemma 4 models are pulled.[/yellow]")
            return None