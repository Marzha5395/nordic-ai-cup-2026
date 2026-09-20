import re
import difflib
import time
from faster_whisper import WhisperModel
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv('data/question_train.csv')

model_id = "Qwen/Qwen3.5-9B"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    token=token
)

audio_model = WhisperModel('small.en', device='cuda', compute_type='float16')  # load at import time


def transcribe(path='data/audio/', file='conversation_sample_77.mp3'):
    start_time = time.perf_counter()

    file_path = path + file
    
    segments, _ = audio_model.transcribe(file_path, language='en', vad_filter=True, word_timestamps=True)
    segments = list(segments)
    results = [{'start': float(s.start), 'end': float(s.end), 'text': s.text} for s in segments]
    words = [(float(w.start), float(w.end), w.word) for s in segments for w in s.words]
    full_text = ''.join([r['text'] for r in results]).strip()
    
    # 2. Stop the timer and calculate duration
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"⏱️ Transcription completed in {execution_time:.2f} seconds.")
    return results, full_text, words


def generate(results, questions):
    start_time = time.perf_counter()

    num_questions = len(questions)
    lines = ''.join(r['text'].strip() for r in results).strip()

    prompt = f"""
    {lines}
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

    # Before you start search for the passages, you first need to merge some of the passages in the conversation, since transcription might not be accurate.
    # For example:
    # "And then the anti-inflammatory gave me reflux on top.
    # Which is why you are on pantoprazole for that."
    # This should be rewritten as:
    # "And then the anti-inflammatory gave me reflux on top, which is why you are on pantoprazole for that."
    # since this is obviously one single sentence. However, do not split a passage into multiple passages, and do not return any of this to me.

def segment(results, words, question):
    lines = ' '.join(r['text'].strip() for r in results).strip()
    
    prompt = f"""
    You are given a question related to a conversation. Your task is to find a single sentence in the conversation that is the evidence for the answer to the question.
    
    This is the transcribed conversation sentences:
    {lines}

    Here is the question related to the conversation.
    {question}

    The answer to this question is yes. Quote, word for word, one sentence of the conversation that establishes this.

    Start by analyzing key words of the question and check which sentence of the conversation best matches the key words and answers the question.
    Example: Are the patient's asthma findings stable at this visit?
    In the example, a valid key word would be "stable", since it is not used that often in random sentences.

    Find the sentence that is evidence of the question.
    Examples of good evidence sentences:
    Question1: Did the patient attend for an annual asthma follow-up?
    Evidence1: "So, this is your annual follow-up."
    Question2: Is the heart examination without abnormal findings?
    Evidence2: "Nothing abnormal to report."

    Note that you are supposed the find support for the answer, not the question itself.
    Answer with one line with only one single sentence from the conversation, that contains evidence of the related question. It must be one single sentence, not more.
    Make sure to have at only one single sentence from the original conversation, it is very important, abosolutely nothing more than that. A sentence should end with a dot, question mark or exclamation mark, and have only commas in between.
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

    # Locate every run of transcribed words that matches the quote, then keep only the
    # longest run of consecutive matches (a quote the model stitched together from separate
    # places, or padded with extra text, must not stretch the span across everything in between)
    quote = [w for w in (normalize(w) for w in response.split()) if w]
    transcript = [normalize(w[2]) for w in words]
    matcher = difflib.SequenceMatcher(None, transcript, quote, autojunk=False)
    blocks = [block for block in matcher.get_matching_blocks() if block.size]
    if not blocks:
        return None, None
    clusters = [[blocks[0]]]
    for block in blocks[1:]:
        previous = clusters[-1][-1]
        if block.a - (previous.a + previous.size) <= 3:
            clusters[-1].append(block)
        else:
            clusters.append([block])
    best = max(clusters, key=lambda c: sum(block.size for block in c))
    start = words[best[0].a][0]
    end = words[best[-1].a + best[-1].size - 1][1]
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
        print()
        print(f'transcriptions: {results}')
        print(f'{evidence_start = }')
        print(f'{evidence_end = }')
        print()

if __name__ == '__main__':
    main()
