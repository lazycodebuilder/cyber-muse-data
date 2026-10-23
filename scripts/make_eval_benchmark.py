"""Generate the 50-question SEALED evaluation benchmark.

These questions are the held-out test set: they are NEVER fed into training.
Training uses data/training_dataset.jsonl only. Keep this file sealed —
do not train on it, or the benchmark scores become meaningless.
"""
import os
import json
import time
import requests

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-v3.2"
URL = "https://openrouter.ai/api/v1/chat/completions"
OUTPUT_FILE = "benchmark/eval_questions.json"

# 50 questions across the guide's 9 benchmark areas
CATEGORIES = [
    ("networking", 6),
    ("web security", 6),
    ("Linux and system security", 6),
    ("secure coding", 6),
    ("cryptography and authentication", 6),
    ("incident response", 5),
    ("forensics and malware analysis", 5),
    ("cloud and container security", 5),
    ("mixed reasoning", 5),
]

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def generate_category(category, count):
    prompt = f"""
You are creating a HELD-OUT evaluation benchmark for a cybersecurity language model.
These questions will NEVER be used for training — they only measure the model.

TOPIC: {category}

Generate exactly {count} UNIQUE, high-quality cybersecurity questions.
Difficulty mix: ~20% beginner, ~50% intermediate, ~30% advanced.

Include a mixture of: conceptual reasoning, troubleshooting, defensive
investigation, diagnosis, code/config/log analysis, realistic scenarios.

STRICT RULES:
1. Every technical premise must be accurate — do not invent commands,
   protocol fields, config options, or OS behavior.
2. Prefer reasoning and diagnosis over simple definitions.
3. Every scenario must contain enough information to answer it.
4. Offensive-security material must be framed as authorized testing,
   CTF/lab work, security research, analysis, or defense.
5. Return ONLY valid JSON, no other text:

[
  {{
    "category": "{category}",
    "difficulty": "intermediate",
    "question": "..."
  }}
]
"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 4000,
        "reasoning": {"enabled": False},
    }
    for attempt in range(1, 4):
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=180)
            if not r.ok:
                print("HTTP", r.status_code, r.text[:200])
                time.sleep(attempt * 2)
                continue
            return parse_json_response(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            print("Attempt failed:", e)
            time.sleep(attempt * 2)
    raise SystemExit(f"FAILED category: {category} — rerun later")


if os.path.exists(OUTPUT_FILE):
    print(OUTPUT_FILE, "already exists — sealed benchmark kept, nothing regenerated.")
    raise SystemExit(0)

all_questions = []
qid = 0
for category, count in CATEGORIES:
    questions = generate_category(category, count)
    for q in questions:
        qid += 1
        q["id"] = qid
        q["category"] = category
        all_questions.append(q)
    print(f"[{category}] +{len(questions)} -> total {len(all_questions)}")
    time.sleep(1)

with open(OUTPUT_FILE, "w") as f:
    json.dump(all_questions, f, indent=2)
print("WROTE", OUTPUT_FILE, "questions:", len(all_questions))
print("SEALED — never use this file for training.")
