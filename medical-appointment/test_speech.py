import numpy as np

from solver import Word
from speech import repair_word_times


def test_leading_pause_is_not_part_of_a_long_word():
    audio = np.zeros(3 * 16000, dtype=np.float32)
    audio[int(1.7 * 16000):2 * 16000] = 0.1
    word = repair_word_times([Word(0.2, 2.0, 'Yes.')], audio)[0]
    assert 1.64 <= word.start <= 1.72
    assert word.end == 2.0


def test_a_normal_short_word_keeps_its_alignment():
    audio = np.ones(16000, dtype=np.float32) * 0.1
    words = [Word(0.2, 0.5, 'Normal.')]
    assert repair_word_times(words, audio) == words


def test_trailing_pause_is_not_part_of_a_long_word():
    audio = np.zeros(2 * 16000, dtype=np.float32)
    audio[:int(0.3 * 16000)] = 0.1
    word = repair_word_times([Word(0.0, 1.5, 'Yes.')], audio)[0]
    assert word.start == 0.0
    assert 0.3 <= word.end <= 0.36


def test_all_silence_does_not_invent_word_boundaries():
    words = [Word(0.1, 1.5, 'Hello.')]
    assert repair_word_times(words, np.zeros(32000, dtype=np.float32)) == words
