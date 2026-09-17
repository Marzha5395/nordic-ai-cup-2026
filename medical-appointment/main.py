# import numpy as np

# audio_bytes = open('data/audio/conversation_sample_4.mp3', 'rb').read()
# audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
# audio_float32 = audio_array.astype(np.float32) / 32768.0

# print(audio_float32.shape)

import time
from faster_whisper import WhisperModel

MODEL = WhisperModel('small.en', device='cpu', compute_type='int8')  # load at import time

def transcribe():
    # 1. Start the timer before transcription begins
    start_time = time.perf_counter()
    
    segments, _ = MODEL.transcribe('data/audio/conversation_sample_4.mp3', language='en', vad_filter=True)
    results = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
    
    # 2. Stop the timer and calculate duration
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"⏱️ Transcription completed in {execution_time:.2f} seconds.")
    return results

print(transcribe())
