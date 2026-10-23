# CyberQwen vs Heretic paired judge.
# The source guide only sketched this file ("same structure as compare_models.py"),
# so this is a full implementation following that structure exactly:
# randomize Cyber/Heretic as A/B, same DeepSeek judge, same rubric,
# map A/B back to real labels, average the scores.
import os
import json
import random
import time
import requests
from statistics import mean

# Judge via Groq (free tier, gpt-oss-120b). Keys from GROQ_API_KEYS env (comma-separated), round-robin.
import itertools
_GROQ_KEYS = [k.strip() for k in os.environ["GROQ_API_KEYS"].split(",") if k.strip()]
assert _GROQ_KEYS, "set GROQ_API_KEYS env var (comma-separated Groq keys)"
_key_cycle = itertools.cycle(_GROQ_KEYS)
MODEL = "openai/gpt-oss-120b"
URL = "https://api.groq.com/openai/v1/chat/completions"
CYBER_FILE = "benchmark/cyber_results.json"
HERETIC_FILE = "benchmark/heretic_results.json"
OUTPUT_FILE = "benchmark/heretic_comparison_scores.json"
MAX_RETRIES = 3

with open(CYBER_FILE, "r") as f:
    cyber_results = json.load(f)
with open(HERETIC_FILE, "r") as f:
    heretic_results = json.load(f)

assert len(cyber_results) == len(heretic_results)

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r") as f:
        scored = json.load(f)
else:
    scored = []

completed_ids = {x["id"] for x in scored}


def parse_json(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def judge(question, answer_a, answer_b):
    prompt = f"""
You are evaluating two AI-generated cybersecurity answers.
You do NOT know which model generated either answer.

QUESTION:

{question}

ANSWER A:

{answer_a}

ANSWER B:

{answer_b}

Score EACH answer from 0 to 5 on:
1. technical_correctness
2. completeness
3. reasoning_quality
4. practical_usefulness

Important:
- do not reward verbosity
- penalize hallucinated facts, commands, protocol behavior,
  configuration settings, or security mechanisms
- do not penalize concise answers if complete
- use the full 0-5 scale
- judge only the supplied answer
- choose overall winner: A, B, or tie

Return ONLY valid JSON:

{{
  "A": {{
    "technical_correctness": 0,
    "completeness": 0,
    "reasoning_quality": 0,
    "practical_usefulness": 0
  }},
  "B": {{
    "technical_correctness": 0,
    "completeness": 0,
    "reasoning_quality": 0,
    "practical_usefulness": 0
  }},
  "winner": "A",
  "reason": "short explanation"
}}

"""
    payload = {
        "model": MODEL,
        "reasoning_effort": "low",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1200,
    }
    for attempt in range(1, MAX_RETRIES + 1):
        headers = {
            "Authorization": f"Bearer {next(_key_cycle)}",
            "Content-Type": "application/json",
        }
        try:
            r = requests.post(
                URL,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if not r.ok:
                time.sleep(attempt * 2)
                continue
            return parse_json(
                r.json()["choices"][0]["message"]["content"]
            )
        except Exception as e:
            print("Attempt failed:", e)
            time.sleep(attempt * 2)
    return None


for i, (cyber, heretic) in enumerate(
    zip(cyber_results, heretic_results),
    start=1,
):
    qid = cyber.get("id", i)
    if qid in completed_ids:
        continue
    if random.random() < 0.5:
        answer_a = cyber["answer"]
        answer_b = heretic["answer"]
        mapping = {"A": "cyber", "B": "heretic"}
    else:
        answer_a = heretic["answer"]
        answer_b = cyber["answer"]
        mapping = {"A": "heretic", "B": "cyber"}
    result = judge(
        cyber["question"],
        answer_a,
        answer_b,
    )
    if result is None:
        continue
    if mapping["A"] == "cyber":
        cyber_score = result["A"]
        heretic_score = result["B"]
    else:
        cyber_score = result["B"]
        heretic_score = result["A"]
    winner_raw = result.get("winner", "tie")
    winner = "tie" if winner_raw == "tie" else mapping[winner_raw]
    scored.append({
        "id": qid,
        "category": cyber.get("category", ""),
        "difficulty": cyber.get("difficulty", ""),
        "question": cyber["question"],
        "cyber_scores": cyber_score,
        "heretic_scores": heretic_score,
        "winner": winner,
        "judge_reason": result.get("reason", ""),
    })
    with open(OUTPUT_FILE, "w") as f:
        json.dump(scored, f, indent=2)

metrics = [
    "technical_correctness",
    "completeness",
    "reasoning_quality",
    "practical_usefulness",
]
cyber_total = []
heretic_total = []
for metric in metrics:
    c = mean(x["cyber_scores"][metric] for x in scored)
    h = mean(x["heretic_scores"][metric] for x in scored)
    cyber_total.append(c)
    heretic_total.append(h)
    print(metric)
    print("Cyber  :", round(c, 3))
    print("Heretic:", round(h, 3))
    print("Delta  :", round(h - c, 3))
    print()

overall_cyber = mean(cyber_total)
overall_heretic = mean(heretic_total)
print("OVERALL CYBER  :", round(overall_cyber, 3), "/ 5")
print("OVERALL HERETIC:", round(overall_heretic, 3), "/ 5")
print(
    "RELATIVE CHANGE:",
    f"{(overall_heretic - overall_cyber) / overall_cyber * 100:.2f}%"
)
print(
    "Head-to-head:",
    "Cyber", sum(x["winner"] == "cyber" for x in scored),
    "Heretic", sum(x["winner"] == "heretic" for x in scored),
    "Ties", sum(x["winner"] == "tie" for x in scored),
)
