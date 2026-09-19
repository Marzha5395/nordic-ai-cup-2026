from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from speech import SpeechRecognizer, default_device


@pytest.mark.parametrize('memory,expected', [('2048', 'cpu'), ('16384', 'cuda')])
def test_device_selection_respects_gpu_capacity(memory, expected):
    default_device.cache_clear()
    with patch('speech.ctranslate2.get_cuda_device_count', return_value=1):
        with patch('speech.subprocess.run', return_value=SimpleNamespace(stdout=memory)):
            assert default_device() == expected
    default_device.cache_clear()


def test_cpu_without_cuda():
    default_device.cache_clear()
    with patch('speech.ctranslate2.get_cuda_device_count', return_value=0):
        assert default_device() == 'cpu'
    default_device.cache_clear()


def test_transcription_is_in_memory_and_preserves_word_timings(monkeypatch, tmp_path):
    monkeypatch.setenv('ASR_DEVICE', 'cpu')
    monkeypatch.setenv('ASR_MODEL', str(tmp_path))
    monkeypatch.setenv('ASR_BATCH_SIZE', '0')
    waveform = np.zeros(16000, dtype=np.float32)
    word = SimpleNamespace(start=0.2, end=0.7, word=' Hello.')
    with patch('speech.WhisperModel') as factory, patch('speech.decode_audio', return_value=waveform):
        factory.return_value.transcribe.return_value = (iter([SimpleNamespace(words=[word])]), None)
        asr = SpeechRecognizer()
        result = asr.transcribe(b'audio bytes')
        assert factory.call_args.kwargs['compute_type'] == 'int8'
        assert factory.call_args.kwargs['local_files_only'] is True
        assert factory.return_value.transcribe.call_args.args[0] is waveform
        assert result.words[0].start == 0.2
        assert result.words[0].end == 0.7
        assert result.words[0].text == 'Hello.'
