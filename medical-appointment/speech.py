import io
import logging
import os
import subprocess
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import ctranslate2
import numpy as np
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=1)
def default_device():
    if not ctranslate2.get_cuda_device_count():
        return 'cpu'
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=5)
        if max(int(value) for value in result.stdout.split()) < 8000:
            return 'cpu'
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 'cuda'


def repair_word_times(words, audio, sample_rate=16000):
    hop = sample_rate // 100
    count = len(audio) // hop
    if not count:
        return words
    rms = np.sqrt(np.mean(np.square(audio[:count * hop].reshape(count, hop)), axis=1))
    threshold = max(0.0001, float(np.percentile(rms, 85)) * 0.015)
    active = np.convolve((rms > threshold).astype(np.int32), np.ones(5, dtype=np.int32), mode='same') > 0
    result = []
    for word in words:
        start, end = word.start, word.end
        if end - start >= 0.7:
            a, b = max(0, int(start * 100)), min(count, int(end * 100))
            window = active[a:b]
            voiced = np.flatnonzero(window)
            if len(voiced):
                silence = ~window
                edges = np.flatnonzero(np.diff(np.r_[False, silence, False])).reshape(-1, 2)
                gaps = [right for left, right in edges if right - left >= 12 and right <= voiced[-1]]
                if gaps:
                    start = max(start, (a + gaps[-1]) / 100 - 0.01)
                if len(window) - voiced[-1] >= 12:
                    end = min(end, (a + voiced[-1] + 1) / 100 + 0.01)
        result.append(replace(word, start=round(start, 3), end=round(max(start, end), 3)))
    return result


class SpeechRecognizer:
    def __init__(self):
        self.device = os.getenv('ASR_DEVICE') or default_device()
        model_path = Path(os.getenv('ASR_MODEL', str(ROOT / 'models' / 'faster-whisper-small.en')))
        if not model_path.exists():
            raise FileNotFoundError(f'ASR model missing at {model_path}. Run prepare_models.py first.')
        self.model = WhisperModel(
            str(model_path), device=self.device,
            compute_type=os.getenv('ASR_COMPUTE_TYPE', 'float16' if self.device == 'cuda' else 'int8'),
            cpu_threads=int(os.getenv('ASR_THREADS', '4')), local_files_only=True,
        )
        self.batch_size = int(os.getenv('ASR_BATCH_SIZE', '0'))
        self.batched = BatchedInferencePipeline(self.model) if self.batch_size else None

    def transcribe(self, audio_bytes):
        from solver import Transcript, Word
        started = time.monotonic()
        audio = decode_audio(io.BytesIO(audio_bytes))
        options = dict(language='en', beam_size=int(os.getenv('ASR_BEAM_SIZE', '5')),
                       word_timestamps=True, vad_filter=True,
                       vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
                       condition_on_previous_text=False)
        if self.batched:
            segments, _ = self.batched.transcribe(audio, batch_size=self.batch_size, **options)
        else:
            segments, _ = self.model.transcribe(audio, **options)
        words = [Word(float(word.start), float(word.end), word.word.strip())
                 for segment in segments for word in (segment.words or [])]
        transcript = Transcript(words)
        if os.getenv('REPAIR_WORD_TIMES', '1') == '1':
            transcript.words = repair_word_times(transcript.words, audio)
        logger.info('ASR: %d words in %.2fs', len(words), time.monotonic() - started)
        return transcript

    def warmup(self):
        import numpy as np
        segments, _ = self.model.transcribe(np.zeros(16000, dtype=np.float32), language='en', beam_size=1)
        list(segments)
