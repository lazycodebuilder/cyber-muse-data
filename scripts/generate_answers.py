import os
import json
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-v3.2"
URL = "https://openrouter.ai/api/v1/chat/completions"
INPUT_FILE = "data/questions_clean.json"
OUTPUT_FILE = "data/training_dataset.jsonl"
BATCH_SIZE = 5
MAX_WORKERS = 4
MAX_RETRIES = 3

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

with open(INPUT_FILE, "r") as f:
    questions = json.load(f)

completed_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    completed_ids.add(json.loads(line)["id"])
                except Exception:
                    pass

remaining = [q for q in questions if q["id"] not in completed_ids]

write_lock = threading.Lock()


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def generate_answers(batch):
    simplified = [
        {
            "id": q["id"],
            "category": q["category"],
            "difficulty": q.get("difficulty", ""),
            "type": q.get("type", ""),
            "question": q["question"],
        }
        for q in batch
    ]
    prompt = f"""
You are an expert cybersecurity instructor creating high-quality
supervised fine-tuning data for a smaller language model.

QUESTIONS:

{json.dumps(simplified, indent=2)}

For EACH question:
- provide a technically accurate answer
- explain enough reasoning for the student model to learn
- be concise and information-dense
- prefer practical explanations over vague textbook wording
- include commands/code/config only when useful
- do not invent technical facts
- do not include hidden chain-of-thought
- target roughly 150-450 words unless less is enough

Return ONLY valid JSON:

[
  {{
    "id": 1,
    "answer": "..."
  }}
]

"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 6000,
        "reasoning": {"enabled": False},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                URL,
                headers=headers,
                json=payload,
                timeout=240,
            )
            if not response.ok:
                print(response.status_code, response.text[:300])
                time.sleep(attempt * 2)
                continue
            return parse_json_response(
                response.json()["choices"][0]["message"]["content"]
            )
        except Exception as e:
            print("Attempt failed:", e)
            time.sleep(attempt * 2)
    return None


batches = [
    remaining[i:i + BATCH_SIZE]
    for i in range(0, len(remaining), BATCH_SIZE)
]

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(generate_answers, batch): batch
        for batch in batches
    }
    for future in as_completed(future_map):
        batch = future_map[future]
        answers = future.result()
        if answers is None:
            continue
        answer_by_id = {
            a["id"]: a["answer"]
            for a in answers
            if "id" in a and "answer" in a
        }
        lines = []
        for q in batch:
            answer = answer_by_id.get(q["id"])
            if not answer:
                continue
            item = {
                "id": q["id"],
                "category": q["category"],
                "difficulty": q.get("difficulty", ""),
                "type": q.get("type", ""),
                "messages": [
                    {"role": "user", "content": q["question"]},
                    {"role": "assistant", "content": answer},
                ],
            }
            lines.append(json.dumps(item, ensure_ascii=False))
        with write_lock:
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
        print(f"[OK] batch done | wrote {len(lines)} answers")
