from piper import PiperVoice
import wave
import logging
from piper.config import SynthesisConfig
from deep_learning.english_teacher.audio.play_audio import ReadingAudio
from io import BytesIO
import numpy as np
import re
import html
from pathlib import Path

logger = logging.getLogger(__name__)


class TTSPiperLocal:
    def __init__(self):
        SPEECH_DIR = Path(__file__).resolve().parent
        AUDIO_DIR = SPEECH_DIR / "files_piper_audio"

        self.local_voice = AUDIO_DIR / "en_GB-cori-medium.onnx"
        self.voice = PiperVoice.load(str(self.local_voice))
        self.syn_config = SynthesisConfig(
            length_scale=1.0,  # Audio speed
            noise_scale=0.667,
            noise_w_scale=0.8,
            volume=1.0,
            normalize_audio=True
        )

    def speak(self, text):
        logger.info("AI synthesizing text to memory...")

        # Cleaning
        text = self.clean_text_for_tts(text)

        # Saving file in memory
        buffer_memoria = BytesIO()

        with wave.open(buffer_memoria, "wb") as wav_file:
            self.voice.synthesize_wav(text, wav_file, syn_config=self.syn_config)

        # Extract binary data
        buffer_memoria.seek(0)

        # Reading a file
        with wave.open(buffer_memoria, "rb") as wav_read:
            read_wav_data = wav_read.readframes(wav_read.getnframes())

        # Convert bytes in matrix (Int16) to sounddevice
        audio_data = np.frombuffer(read_wav_data, dtype=np.int16)
        voice_frequency = self.voice.config.sample_rate

        logger.info("AI speaking directly from RAM")
        read_class = ReadingAudio(audio_data, voice_frequency)
        read_class.reading_file()
        logger.info("speech concluded")

    @staticmethod
    def clean_text_for_tts(text_ai: str) -> str:
        # Decodes HTML entities, if present.
        text = html.unescape(text_ai)

        # Remove code blocks, while keeping their content.
        text = re.sub(r"```[^\n]*\n?(.*?)```", r"\1", text, flags=re.DOTALL)

        # Remove inline code, while keeping the content.
        text = re.sub(r"`([^`]*)`", r"\1", text)

        # Remove Markdown images.
        text = re.sub(r"!\[([^]]*)]\([^)]+\)", r"\1", text)

        # Converts Markdown links into visible text.
        text = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", text)

        # Remove Markdown headings.
        text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)

        # Remove Markdown quotes.
        text = re.sub(r"(?m)^\s*>\s?", "", text)

        # Remove list markers from the beginning of lines.
        text = re.sub(r"(?m)^\s*[-+*]\s+", "", text)
        text = re.sub(r"(?m)^\s*\d+[.)]\s+", "", text)

        # Remove bold and italics: **text**, __text__, *text*, _text_.
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text, flags=re.DOTALL)
        text = re.sub(r"(?<!\w)([*_])(.*?)\1(?!\w)", r"\2", text, flags=re.DOTALL)

        # Remove any remaining formatting symbols.
        text = re.sub(r"[*_~#]", "", text)

        # Reduces repeated spaces without removing the breaks between paragraphs.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()