import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "./Qwen3-4B-Instruct-2507"
ADAPTER = "./models/cyber-muse-lora"
QUESTIONS_FILE = "benchmark/eval_questions.json"
OUTPUT_FILE = "benchmark/cyber_results.json"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token_id

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map={"": 0},
    trust_remote_code=True,
)
print("Loading Cyber-Muse LoRA adapter...")
model = PeftModel.from_pretrained(
    model,
    ADAPTER,
)
model.eval()

with open(QUESTIONS_FILE, "r") as f:
    questions = json.load(f)

results = []
for i, q in enumerate(questions, start=1):
    print(f"[{i}/{len(questions)}] {q['question'][:80]}...")
    messages = [
        {"role": "user", "content": q["question"]}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()
    results.append({
        "id": q.get("id", i),
        "category": q.get("category", ""),
        "difficulty": q.get("difficulty", ""),
        "question": q["question"],
        "answer": answer,
    })
    print(answer[:300])
    print("-" * 80)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)
print("Saved:", OUTPUT_FILE)
