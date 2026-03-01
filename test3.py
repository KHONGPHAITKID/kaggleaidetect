import torch
import os
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from collections import Counter
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    login(token=HF_TOKEN)

# using dagshub for data versioning and mlflow for experiment tracking
# import dagshub
# import mlflow
# dagshub.init(repo_owner='nghessss', repo_name='semivalA', mlflow=True)
# with mlflow.start_run():
#   # Your training code here...
#   mlflow.log_metric('accuracy', 42)
#   mlflow.log_param('Param name', 'Value')

# --- CONFIGURATION ---
# Gemma-2-9B-It is NOT gated and fits perfectly in 12GB VRAM
model_id = "google/gemma-2-9b-it"
# More option: 
model_list = [
    "google/gemma-2-9b-it",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "tinywell/THUDM-glm-4-9b-chat-4bit",
    "microsoft/phi-4-mini-instruct"
]


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto", 
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    token=HF_TOKEN or None
)

def detect(code):
    # Prompt tuned for Gemma's instruction format
    # We explicitly tell it to look for 'AI-style' patterns to fix your previous 0% bias
    prompt = (
        "Instructions: You are a code auditor. You must identify if code is AI-generated (1) or Human-written (0).\n"
        "AI code usually has: Perfect indentation, standard naming (e.g., 'result', 'data'), and lack of 'hacks'.\n"
        "Human code usually has: Inconsistent spacing, unique variable names, or non-standard logic.\n\n"
        """
        Given the example 1:
        "code":"import sys\ninf = float('inf')\nfrom bisect import bisect_left, bisect_right\n\ndef get_array():\n\treturn list(map(int, sys.stdin.readline().strip().split()))\n\ndef get_ints():\n\treturn map(int, sys.stdin.readline().strip().split())\n\ndef input():\n\treturn sys.stdin.readline().strip()\nn = int(input())\nArr = get_array()\ntotal = sum(Arr)\nalice = Arr[0]\nct = 0\nstore = alice\nindex = [1]\nfor i in range(1, n):\n\tif Arr[i] <= int(alice / 2):\n\t\tct += 1\n\t\tstore += Arr[i]\n\t\tindex.append(i + 1)\nif store <= total // 2:\n\tprint(0)\n\texit()\nif ct == 0:\n\tprint(1)\n\tprint(1)\n\texit()\nelse:\n\tprint(ct + 1)\n\tprint(*index)\n"
        "label": 0
        Given the example 2:
        "code":"def Range(a): return min(max(a, -10), 10)\n\ndef FirstProb(n):\nreturn (0, 0, 0)\n\n\ndef Escalar(a1, a2):\nreturn sum(x - y for x, y in zip(a1, a2))\n\n'''\n}\n\n\ndef main():\nprint('is idle body') if Sum(FirstProb(int(stdin.readline()))) == 0 else print('is not idle body')\n\n\nif __name__ == '__main__': main()" 
        "label":1
        """
        f"Code to evaluate:\n```\n{code}\n```\n\n"
        "Answer with ONLY the digit 0 or 1. Response:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=2, 
            temperature=0.01,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(output_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
    
    # Simple logic to handle cases where it might say "0." or "1."
    while ('1' in result) and ('0' in result):
        print("Ambiguous output detected, re-evaluating...")
        return detect(code)
    if '1' in result: return 1
    return 0

# --- DATA PROCESSING ---
try:
    # Testing on 100 rows to check for bias shift
    df = pd.read_parquet('data/train.parquet').head(100)
    results = []

    print(f"--- Running Detection with {model_id} ---")

    for idx, row in df.iterrows():
        true_label = int(row['label'])
        pred_label = detect(row['code'])
        results.append({'true': true_label, 'pred': pred_label})
        
        status = "✅" if true_label == pred_label else "❌"
        # Print progress to see if the "0" bias is still there
        print(f"Row {idx}: True={true_label}, Pred={pred_label} {status}")

    # --- FINAL STATS ---
    res_df = pd.DataFrame(results)
    tp = len(res_df[(res_df['true'] == 1) & (res_df['pred'] == 1)])
    fp = len(res_df[(res_df['true'] == 0) & (res_df['pred'] == 1)])
    fn = len(res_df[(res_df['true'] == 1) & (res_df['pred'] == 0)])
    tn = len(res_df[(res_df['true'] == 0) & (res_df['pred'] == 0)])
    
    accuracy = (tp + tn) / len(res_df)
    
    print("\n" + "="*30)
    print(f"   RESULTS FOR {model_id}")
    print("="*30)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Prediction Ratio: AI={res_df['pred'].mean()*100:.1f}%, Human={(1-res_df['pred'].mean())*100:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"TP (Caught AI): {tp} | FP (False Alarm): {fp}")
    print(f"FN (Missed AI): {fn} | TN (Correct Human): {tn}")
    print("="*30)

except Exception as e:
    print(f"An error occurred: {e}")
