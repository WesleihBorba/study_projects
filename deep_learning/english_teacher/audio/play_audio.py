import sounddevice as sd
import logging

logger = logging.getLogger(__name__)


class ReadingAudio:
    def __init__(self, audio, frequency):
        self.audio_data, self.frequency = audio, frequency

    def reading_file(self):
        logger.info("playing the audio...")
        sd.play(self.audio_data, samplerate=self.frequency)
        sd.wait()