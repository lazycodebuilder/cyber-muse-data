import os
import json
import time
import requests

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = "deepseek/deepseek-v3.2"
URL = "https://openrouter.ai/api/v1/chat/completions"
INPUT_FILE = "data/questions_raw.json"
REVIEW_FILE = "data/questions_reviewed.json"
CLEAN_FILE = "data/questions_clean.json"
BATCH_SIZE = 40
MAX_RETRIES = 3

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

with open(INPUT_FILE, "r") as f:
    raw_questions = json.load(f)

# tolerate a few common generator key mistakes
for q in raw_questions:
    if "question" not in q:
        fallback = q.get("user") or q.get("prompt")
        if fallback:
            q["question"] = fallback

if os.path.exists(REVIEW_FILE):
    with open(REVIEW_FILE, "r") as f:
        reviews = json.load(f)
else:
    reviews = []

reviewed_ids = {x["id"] for x in reviews}
remaining = [
    q for q in raw_questions
    if q.get("id") not in reviewed_ids and q.get("question")
]


def parse_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def validate_batch(batch):
    compact = [
        {
            "id": q["id"],
            "category": q.get("category", ""),
            "question": q["question"],
        }
        for q in batch
    ]
    prompt = f"""
You are technically validating cybersecurity questions for a supervised
fine-tuning dataset.

QUESTIONS:

{json.dumps(compact, indent=2)}

For every question choose exactly one verdict:

accept
- technically sound and answerable as written

rewrite
- contains a real technical/logical flaw but the idea is salvageable

reject
- fundamentally broken, misleading, unanswerable, or based on a false premise

Rules:
- check protocol layers, commands, fields, operating-system behavior,
  configuration assumptions, cloud/IAM behavior, security terminology,
  and scenario consistency
- do not rewrite merely for style
- if rewriting, preserve the original learning objective
- no essays or long explanations
- output exactly one result per input ID
- valid JSON only

Return:

[
  {{
    "id": 1,
    "verdict": "accept",
    "corrected_question": null
  }}
]

"""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 3500,
        "reasoning": {"enabled": False},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                URL,
                headers=headers,
                json=payload,
                timeout=240,
            )
            if not r.ok:
                print("HTTP", r.status_code, r.text[:300])
                time.sleep(attempt * 2)
                continue
            result = parse_json(
                r.json()["choices"][0]["message"]["content"]
            )
            return result
        except Exception as e:
            print("Attempt failed:", e)
            time.sleep(attempt * 2)
    return None


batches = [
    remaining[i:i + BATCH_SIZE]
    for i in range(0, len(remaining), BATCH_SIZE)
]

for batch_index, batch in enumerate(batches, start=1):
    print(f"Validating batch {batch_index}")
    result = validate_batch(batch)
    if result is None:
        print("Batch failed. Rerun later; resume support will skip completed IDs.")
        continue
    reviews.extend(result)
    with open(REVIEW_FILE, "w") as f:
        json.dump(reviews, f, indent=2)
    counts = {"accept": 0, "rewrite": 0, "reject": 0}
    for x in result:
        verdict = x.get("verdict", "").lower()
        if verdict in counts:
            counts[verdict] += 1
    print(
        f"ACCEPT: {counts['accept']} | "
        f"REWRITE: {counts['rewrite']} | "
        f"REJECT: {counts['reject']}"
    )
    print("Total reviewed:", len(reviews))

review_by_id = {
    r["id"]: r
    for r in reviews
}

clean = []
for q in raw_questions:
    review = review_by_id.get(q.get("id"))
    if not review:
        continue
    verdict = review.get("verdict", "").lower()
    if verdict == "reject":
        continue
    new_q = dict(q)
    if verdict == "rewrite":
        corrected = review.get("corrected_question")
        if corrected:
            new_q["original_question"] = q["question"]
            new_q["question"] = corrected
    clean.append(new_q)

with open(CLEAN_FILE, "w") as f:
    json.dump(clean, f, indent=2)
print("Clean questions:", len(clean))
print("Saved:", CLEAN_FILE)
