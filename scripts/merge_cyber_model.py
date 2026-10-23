import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "./Qwen3-4B-Instruct-2507"
ADAPTER = "./models/cyber-muse-lora"
OUTPUT = "./models/cyber-muse-merged"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, ADAPTER)
model = model.merge_and_unload()
model.save_pretrained(
    OUTPUT,
    safe_serialization=True,
    max_shard_size="4GB",
)
tokenizer.save_pretrained(OUTPUT)
print("Saved:", OUTPUT)
