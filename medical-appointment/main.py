import re
import difflib
import time
from faster_whisper import WhisperModel
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv('data/question_train.csv')

model_id = "Qwen/Qwen3.5-9B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

audio_model = WhisperModel('small.en', device='cuda', compute_type='float16')  # load at import time


def transcribe(path='data/audio/', file='conversation_sample_77.mp3'):
    start_time = time.perf_counter()

    file_path = path + file
    
    segments, _ = audio_model.transcribe(file_path, language='en', vad_filter=True, word_timestamps=True)
    segments = list(segments)
    results = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
    words = [(w.start, w.end, w.word) for s in segments for w in s.words]
    full_text = ''.join([r['text'] for r in results]).strip()
    
    # 2. Stop the timer and calculate duration
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"⏱️ Transcription completed in {execution_time:.2f} seconds.")
    return results, full_text, words


def generate(results, questions):
    start_time = time.perf_counter()

    num_questions = len(questions)

    prompt = f"""
    {results}
    The above text is a conversation. Now answer the following questions about the conversation with yes or no. Answer with one line for each question, each line containing only one word: yes or no.
    {'\n'.join(questions)}
    Make sure to answer with the same amount of rows as there are questions. There should be a total of {len(questions)} questions, one on each row.
    Be careful when answering the questions as there are some hard negatives, meaning most of the information given is correct but one detail is off.
    """

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    generate_ids = model.generate(
        **inputs, 
        max_new_tokens=300
    )

    # Decode response while slicing out the original prompt tokens
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    predictions = response.lower().strip().split('\n')

    print(f"⏱️ Generation completed in {execution_time:.2f} seconds.")
    return predictions[:num_questions]

def normalize(word):
    return re.sub(r'[^a-z0-9]', '', word.lower())


def segment(results, words, question):
    lines = '\n'.join(r['text'].strip() for r in results)
    prompt = f"""
    You are given a question related to a conversation. Your task is to find the passage of the conversation that is the evidence for the answer to the question.
    This is the conversation:
    {lines}
    Here is the question related to the conversation.
    {question}
    The answer to this question is yes. Quote, word for word, the shortest passage of the conversation that establishes this. Usually this is a single sentence or part of a sentence.
    Examples of good quotes:
    Question: Is the heart examination without abnormal findings? Quote: Nothing abnormal to report.
    Question: Was muscle tension found in the neck and shoulders? Quote: I can feel muscle tension in your neck and shoulders.
    Question: Did the patient ask for penicillin? Quote: I want penicillin for it.
    Question: Will the patient take Fluconazole 50 mg? Quote: And fluconazole, 50 milligrams for seven days, for the mouth.
    Note that you are supposed the find support for the answer, not the question itself.
    Answer with the quote only.
    """

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    # Decode response while slicing out the original prompt tokens
    response = tokenizer.decode(generate_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    # Locate the quote among the transcribed words: slide a window of the quote's length
    # and keep the one sharing the most words with it
    quote = [w for w in (normalize(w) for w in response.split()) if w]
    transcript = [normalize(w[2]) for w in words]
    n = len(quote)
    best_matches, i = -1, 0
    for k in range(max(1, len(transcript) - n + 1)):
        matcher = difflib.SequenceMatcher(None, transcript[k:k + n], quote, autojunk=False)
        matches = sum(block.size for block in matcher.get_matching_blocks())
        if matches > best_matches:
            best_matches, i = matches, k

    # Trim the window to the first and last word that actually match the quote
    matcher = difflib.SequenceMatcher(None, transcript[i:i + n], quote, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return None, None
    start = words[i + blocks[0].a][0]
    end = words[i + blocks[-1].a + blocks[-1].size - 1][1]
    return start, end


def main():
    for id in df['transcript_id'].unique():
        file_name='conversation_' + id + '.mp3'
        results, full_text, words = transcribe(file=file_name)

        df_sample = df[df['transcript_id'] == id]
        questions = df_sample['question'].to_list()

        predictions = generate(results, questions)
        answers = df_sample['answer'].to_list()

        n_correct = sum(predictions[i] == answers[i] for i in range(max(len(predictions), len(answers))))
        print(f'Sample id: {id} | Correct predictions: {n_correct}/10')

        evidence_start = [0]*len(questions)
        evidence_end = [60]*len(questions)
        predictions = [p == 'yes' for p in predictions]
        for i in range(len(questions)):
            if predictions[i]:
                start, end = segment(results, words, questions[i])
                evidence_start[i] = start
                evidence_end[i] = end

if __name__ == '__main__':
    main()
