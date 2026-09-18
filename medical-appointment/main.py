import time
from faster_whisper import WhisperModel
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

df = pd.read_csv('data/question_train.csv')

model_id = "Qwen/Qwen3.5-2B"

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
    
    segments, _ = audio_model.transcribe(file_path, language='en', vad_filter=False)
    results = [{'start': s.start, 'end': s.end, 'text': s.text} for s in segments]
    full_text = ''.join([r['text'] for r in results]).strip()
    
    # 2. Stop the timer and calculate duration
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    print(f"⏱️ Transcription completed in {execution_time:.2f} seconds.")
    return results, full_text


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
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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
    predictions = response.lower().split()

    print(f"⏱️ Generation completed in {execution_time:.2f} seconds.")
    return predictions[:num_questions]

def main():
    for id in df['transcript_id'].unique():
        file_name='conversation_' + id + '.mp3'
        results, full_text = transcribe(file=file_name)

        df_sample = df[df['transcript_id'] == id]
        questions = df_sample['question'].to_list()

        predictions = generate(results, questions)
        answers = df_sample['answer'].to_list()

        n_correct = sum(predictions[i] == answers[i] for i in range(max(len(predictions), len(answers))))
        print(f'Sample id: {id} | Correct predictions: {n_correct}/10')

if __name__ == '__main__':
    main()
