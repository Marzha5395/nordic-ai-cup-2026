import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use "Qwen/Qwen3.5-9B" (or "Qwen/Qwen2.5-7B-Instruct")
model_id = "Qwen/Qwen3.5-2B"

# Use Auto classes so transformers maps the right architecture automatically
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

prompt = "Hey, explain this python snippet."

# Apply the model's native chat template
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate
generate_ids = model.generate(
    **inputs, 
    max_new_tokens=100
)

# Decode response while slicing out the original prompt tokens
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generate_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
