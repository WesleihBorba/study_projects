import whisper
import logging
import math
import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import io

logger = logging.getLogger(__name__)


class TranscribeAudio:
    def __init__(self):
        self.model = whisper.load_model("small.en")
        self.text_audio = None

    def transcribe_file(self, audio_bytes):
        try:
            audio_data, sample_rate = sf.read(
                io.BytesIO(audio_bytes),
                dtype="float32",
                always_2d=True
            )

            if audio_data.size == 0:
                logger.warning("The uploaded audio is empty.")
                return ""

            audio_data = audio_data.mean(axis=1)

            target_sample_rate = 16000

            if sample_rate != target_sample_rate:
                divisor = math.gcd(sample_rate, target_sample_rate)

                audio_data = resample_poly(
                    audio_data,
                    target_sample_rate // divisor,
                    sample_rate // divisor
                ).astype(np.float32)

            self.text_audio = self.model.transcribe(
                audio_data,
                fp16=False
            )

            text = self.text_audio["text"].strip()

            logger.info("Transcribed audio: %s", text)

            if text:
                return f"New conversation:\n{text}"

            return ""

        except Exception:
            logger.exception("Error processing audio.")
            raise