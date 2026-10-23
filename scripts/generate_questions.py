import os
import json
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-v3.2"
URL = "https://openrouter.ai/api/v1/chat/completions"
TOPICS_FILE = "data/topics.json"
OUTPUT_FILE = "data/questions_raw.json"
TARGET_QUESTIONS = 3000
QUESTIONS_PER_BATCH = 10
BATCHES_PER_TOPIC = 5
MAX_WORKERS = 4
MAX_RETRIES = 3

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

with open(TOPICS_FILE, "r") as f:
    topics = json.load(f)

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        all_questions = json.load(f)
else:
    all_questions = []

topic_counts = {}
for q in all_questions:
    topic = q["category"]
    topic_counts[topic] = topic_counts.get(topic, 0) + 1

save_lock = threading.Lock()


def parse_json_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def generate_batch(topic, batch_number):
    prompt = f"""
You are creating a high-quality cybersecurity instruction-training dataset.

TOPIC:

{topic}

Generate exactly {QUESTIONS_PER_BATCH} UNIQUE cybersecurity questions.

Difficulty distribution:
- 20% beginner
- 50% intermediate
- 30% advanced

Include a mixture of:
- conceptual reasoning
- troubleshooting
- defensive investigation
- network/system diagnosis
- code analysis
- configuration analysis
- log or packet analysis
- realistic security scenarios
- comparisons
- edge cases

STRICT QUALITY REQUIREMENTS:
1. Every technical premise must be accurate.
2. Do not invent commands, protocol fields, configuration options,
   security mechanisms, or operating-system behavior.
3. Do not confuse concepts from different protocol layers.
4. Avoid trivial variations of the same question.
5. Prefer reasoning and diagnosis over simple definitions.
6. Every scenario must contain enough information to meaningfully answer it.
7. If you are uncertain about a technical premise, do NOT include it.
8. Offensive-security material must be framed as authorized testing,
   CTF/lab work, security research, analysis, or defense.
9. Do NOT provide answers.
10. Silently verify every question before including it.

Return ONLY valid JSON:

[
  {{
    "category": "{topic}",
    "difficulty": "beginner",
    "type": "conceptual",
    "question": "..."
  }}
]

"""

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 3500,
        "reasoning": {"enabled": False},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if not response.ok:
                print(response.status_code, response.text[:300])
                time.sleep(attempt * 2)
                continue
            questions = parse_json_response(
                response.json()["choices"][0]["message"]["content"]
            )
            for q in questions:
                q["category"] = topic
                q["batch"] = batch_number + 1
            return topic, batch_number, questions
        except Exception as e:
            print("Attempt failed:", e)
            time.sleep(attempt * 2)
    return topic, batch_number, None


jobs = []
for topic in topics:
    already_generated = topic_counts.get(topic, 0)
    completed_batches = already_generated // QUESTIONS_PER_BATCH
    for batch_number in range(completed_batches, BATCHES_PER_TOPIC):
        jobs.append((topic, batch_number))

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_map = {
        executor.submit(generate_batch, topic, batch_number): (topic, batch_number)
        for topic, batch_number in jobs
    }
    for future in as_completed(future_map):
        topic, batch_number = future_map[future]
        result_topic, result_batch, questions = future.result()
        if questions is None:
            print("[FAILED]", topic, batch_number + 1)
            continue
        with save_lock:
            for q in questions:
                q["id"] = len(all_questions) + 1
                all_questions.append(q)
            with open(OUTPUT_FILE, "w") as f:
                json.dump(all_questions, f, indent=2)
        print(
            f"[OK] {result_topic} batch {result_batch+1} "
            f"-> {len(questions)} questions | total: {len(all_questions)}"
        )

all_questions = all_questions[:TARGET_QUESTIONS]
for i, q in enumerate(all_questions, start=1):
    q["id"] = i
with open(OUTPUT_FILE, "w") as f:
    json.dump(all_questions, f, indent=2)
print("GENERATION COMPLETE:", len(all_questions))
