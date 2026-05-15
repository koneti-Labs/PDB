import os
import time
import keyboard
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from scipy.io.wavfile import read as wavread
from faster_whisper import WhisperModel
import torch
from rich.console import Console
from rich.panel import Panel
from .config import AUDIO_DIR, SAMPLE_RATE, CHANNELS, MAX_RECORD_SECONDS, WHISPER_MODEL, SUPPORTED_LANGUAGES, LANG_ID_MODEL

console = Console()

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

class AudioHandler:
    def __init__(self):
        self.model = None
        self.lang_id_model = None
        self.lang_id_processor = None
        self.temp_file = None
        self.is_recording = False
        self.audio_data = []
        self.device = self._get_device()
        
    def _get_device(self):
        """Safely determine compute device"""
        try:
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        return "cpu"
        
    def load_model(self):
        """Load faster-whisper model (downloads on first run)"""
        if self.model is None:
            console.print("[yellow]Loading Whisper model... This may take a minute on first run.[/yellow]")
            compute_type = "float16" if self.device == "cuda" else "int8"
            
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=self.device,
                compute_type=compute_type
            )
            console.print("[green]✓ Whisper model loaded successfully![/green]")
    
    def load_language_id_model(self):
        """Load AI4Bharat Indic Conformer model for language identification"""
        if self.lang_id_model is None:
            console.print("[yellow]Loading language identification model...[/yellow]")
            try:
                self.lang_id_processor = AutoFeatureExtractor.from_pretrained(LANG_ID_MODEL)
                self.lang_id_model = AutoModelForAudioClassification.from_pretrained(LANG_ID_MODEL)
                self.lang_id_model = self.lang_id_model.to(self.device)
                self.lang_id_model.eval()
                console.print("[green]✓ Language identification model loaded successfully![/green]")
            except Exception as e:
                console.print(f"[red]Failed to load language ID model: {e}[/red]")
                console.print("[yellow]Will fall back to Whisper's language detection[/yellow]")
                self.lang_id_model = None

    def record_push_to_talk(self):
        """Record audio using Push-to-Talk (Hold Spacebar)"""
        console.print("\n[bold cyan]Hold [white on blue] SPACEBAR [/white on blue] while speaking. Release to stop.[/bold cyan]")
        
        self.audio_data = []
        self.is_recording = False

        def callback(indata, frames, time_info, status):
            if status:
                console.print(f"[yellow]Audio warning: {status}[/yellow]")
            if self.is_recording:
                self.audio_data.append(indata.copy())

        # Start audio stream with better error handling
        try:
            # Try to find a working audio device
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE, 
                channels=CHANNELS, 
                dtype='float32', 
                callback=callback,
                latency='low',
                blocksize=0  # Use default blocksize
            )
        except Exception as e:
            console.print(f"[red]✗ Error initializing audio device:[/red] {e}")
            console.print("[yellow]Trying alternative audio configuration...[/yellow]")
            try:
                # Fallback: try without specifying device
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE, 
                    channels=CHANNELS, 
                    dtype='float32', 
                    callback=callback
                )
            except Exception as e2:
                console.print(f"[red]✗ Failed to initialize audio: {e2}[/red]")
                console.print("[yellow]Available audio devices:[/yellow]")
                console.print(sd.query_devices())
                return None
        
        stream.start()

        try:
            console.print("[bold red]🔴 Recording... (Hold Space)[/bold red]")
            self.is_recording = True
            start_time = time.time()
            space_pressed = False

            # Wait for Spacebar to be pressed
            while not space_pressed:
                if keyboard.is_pressed('space'):
                    space_pressed = True
                    console.print("[bold red]🔴 Recording...[/bold red]")
                time.sleep(0.01)

            # Wait until Spacebar is released
            while keyboard.is_pressed('space'):
                if time.time() - start_time > MAX_RECORD_SECONDS:
                    console.print("[yellow]⚠️  Max recording time reached.[/yellow]")
                    break
                time.sleep(0.01)

            self.is_recording = False
            console.print("[green]▶️  Processing...[/green]")

        finally:
            stream.stop()
            stream.close()

        if len(self.audio_data) == 0:
            console.print("[red]No audio recorded.[/red]")
            return None

        # Save to temporary file
        self.temp_file = AUDIO_DIR / f"temp_rec_{int(time.time())}.wav"
        audio_array = np.concatenate(self.audio_data, axis=0)
        write(str(self.temp_file), SAMPLE_RATE, audio_array)
        
        return self.temp_file

    def identify_language(self, audio_path):
        """Identify language using AI4Bharat Indic Conformer model"""
        try:
            if self.lang_id_model is None:
                return None
            
            # Load audio
            audio_array, sr = wavread(str(audio_path))
            
            # Resample if necessary (model expects 16kHz)
            if sr != 16000:
                import librosa
                audio_array = librosa.resample(audio_array.astype(float), orig_sr=sr, target_sr=16000)
            else:
                audio_array = audio_array.astype(float)
            
            # Normalize
            audio_array = audio_array / np.max(np.abs(audio_array))
            
            # Process with feature extractor
            inputs = self.lang_id_processor(
                audio_array, 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            
            # Get predictions
            with torch.no_grad():
                logits = self.lang_id_model(inputs["input_values"].to(self.device)).logits
            
            # Get predicted language
            predicted_class_id = logits.argmax(-1).item()
            predicted_label = self.lang_id_model.config.id2label[predicted_class_id]
            confidence = torch.nn.functional.softmax(logits, dim=-1).max().item()
            
            # Map to supported language codes
            lang_mapping = {
                "eng": "en",
                "hin": "hi",
                "tel": "te",
                "kan": "kn",
                "tam": "ta",
            }
            
            lang_code = lang_mapping.get(predicted_label[:3], None)
            
            if lang_code:
                return {
                    "language_code": lang_code,
                    "language_name": SUPPORTED_LANGUAGES.get(lang_code, "Unknown"),
                    "confidence": confidence
                }
            
            return None
        except Exception as e:
            console.print(f"[yellow]Language ID error: {e}. Using Whisper detection.[/yellow]")
            return None

    def transcribe(self, audio_path):
        """Transcribe with language detection using AI4Bharat"""
        self.load_model()
        console.print("[cyan]Identifying language...[/cyan]")
        
        # Try AI4Bharat first
        self.load_language_id_model()
        lang_result = self.identify_language(audio_path)
        
        if lang_result:
            console.print(f"[green]✓ Detected: {lang_result['language_name']}[/green]")
            lang_code = lang_result["language_code"]
            confidence = lang_result["confidence"]
        else:
            # Fallback to Whisper's detection
            console.print("[yellow]Using Whisper language detection...[/yellow]")
            lang_code = None
            confidence = None
        
        console.print("[cyan]Transcribing...[/cyan]")
        
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=3,
            word_timestamps=False,
            language=lang_code,                # Use detected language or auto-detect
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            temperature=0.0,
        )

        transcription = " ".join(segment.text for segment in segments).strip()
        
        # If AI4Bharat didn't work, use Whisper's detection
        if confidence is None:
            lang_code = info.language
            confidence = info.language_probability
        
        lang_name = SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
        
        console.print("[green]✓ Done![/green]")

        # Immediate cleanup
        self.cleanup_audio(audio_path)

        return {
            "language_code": lang_code,
            "language_name": lang_name,
            "confidence": round(float(confidence), 4),
            "transcription": transcription or "[No speech detected]"
        }

    def cleanup_audio(self, file_path):
        """Mandatory: Delete audio immediately after use"""
        try:
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            console.print(f"[red]Warning: Could not delete audio file: {e}[/red]")