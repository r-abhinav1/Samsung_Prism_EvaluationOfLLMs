import json
import requests
from tqdm import tqdm

# Configurations
API_KEY = 'YOUR_API_KEY'  # replace with your actual key
MODEL_NAME = 'mistral-small'   # or 'mistral-medium', 'mistral-large' if available
INPUT_FILE = 'prompts(draft).json'    # your input file
OUTPUT_FILE = 'prompts_filled.json'  # your output file

# Mistral API endpoint
API_URL = 'https://api.mistral.ai/v1/chat/completions'

# Set up HTTP headers
headers = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

# Function to get answer from Mistral
def get_mistral_answer(prompt):
    payload = {
        'model': MODEL_NAME,
        'messages': [
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.7
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # raises error for bad responses
    result = response.json()
    return result['choices'][0]['message']['content'].strip()

# Load your JSON prompts
with open(INPUT_FILE, 'r') as infile:
    prompts_data = json.load(infile)

# Process each prompt and fill model + answer
for item in tqdm(prompts_data, desc="Processing prompts"):
    item['model'] = MODEL_NAME
    try:
        item['answer'] = get_mistral_answer(item['prompt'])
    except Exception as e:
        print(f"Error for prompt ID {item['id']}: {e}")
        item['answer'] = 'Error generating response.'

# Save filled JSON
with open(OUTPUT_FILE, 'w') as outfile:
    json.dump(prompts_data, outfile, indent=2)

print(f"\nDone! Filled data saved to {OUTPUT_FILE}")
