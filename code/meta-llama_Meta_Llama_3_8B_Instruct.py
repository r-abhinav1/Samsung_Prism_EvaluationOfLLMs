# STEP 1: Install dependencies
!pip install -q transformers accelerate huggingface_hub

# STEP 2: Authenticate with Hugging Face
from huggingface_hub import login
login("")  # Use your token

#  STEP 3: Import libraries
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# STEP 4: Load LLaMA 3 - 8B Instruct model
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
tokenizer.pad_token_id = tokenizer.eos_token_id
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)  # GPU

# STEP 5: Upload prompts(draft).json
from google.colab import files
uploaded = files.upload()

input_path = "prompts(draft).json"
output_path = "prompts_completed_llama3_8b.json"

# STEP 6: Load prompts
with open(input_path, "r", encoding="utf-8") as f:
    original_data = json.load(f)

# STEP 7: Resume if output exists
if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        saved_data = json.load(f)
    print(f"🔄 Resuming with {len(saved_data)} entries.")
else:
    saved_data = original_data

# STEP 8: Helper function
def is_empty_or_error(answer):
    if not answer:
        return True
    if isinstance(answer, str):
        cleaned = answer.strip().lower()
        return cleaned in ["", "[error]"] or cleaned.startswith("write a") or cleaned.startswith("describe")
    return False

remaining = [item for item in saved_data if is_empty_or_error(item.get("answer"))]

def chunk(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]

# STEP 9: Process prompts
batch_size = 1  # 8B model - one prompt at a time
success = 0
errors = 0

for batch in tqdm(chunk(remaining, batch_size), desc="Generating"):
    prompts = [item["prompt"] for item in batch]

    try:
        results = generator(
            prompts,
            max_new_tokens=200,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )

        for idx, item in enumerate(batch):
            try:
                output = results[idx]["generated_text"]
                item["answer"] = output.replace(item["prompt"], "").strip()
                item["model"] = model_id
                success += 1
            except Exception as inner_e:
                item["answer"] = "[ERROR]"
                item["model"] = model_id
                errors += 1

    except Exception as e:
        print("⚠️ Batch error:", e)
        for item in batch:
            item["answer"] = "[ERROR]"
            item["model"] = model_id
            errors += 1

    # Save progress
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(saved_data, f, indent=2, ensure_ascii=False)

# STEP 10: Download results
files.download(output_path)
print(f"DONE — {success} success, {errors} errors.")

