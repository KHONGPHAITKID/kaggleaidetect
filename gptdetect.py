import ast
import random
import string
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_PERTURBATIONS = 10 
TRAIN_SAMPLES = 500  # Samples to find the threshold

print(f"Loading model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to(DEVICE)
model.eval()

# --- LOG-RANK SCORING ---
def get_score(code_snippet):
    inputs = tokenizer(code_snippet, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
    if inputs['input_ids'].shape[1] < 5: return 0.0
    
    with torch.no_grad():
        logits = model(**inputs).logits
        labels = inputs["input_ids"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate Rank
        true_token_logits = shift_logits.gather(-1, shift_labels.unsqueeze(-1))
        rank = (shift_logits > true_token_logits).sum(dim=-1) + 1
        return -torch.log(rank.float()).mean().item()

# --- PERTURBATION ENGINE ---
class AdvancedPerturber(ast.NodeTransformer):
    def __init__(self):
        self.mapping = {}
    def _random_name(self):
        return 'v_' + ''.join(random.choices(string.ascii_lowercase, k=4))
    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load)) and len(node.id) > 1:
            if node.id not in self.mapping: self.mapping[node.id] = self._random_name()
            return ast.copy_location(ast.Name(id=self.mapping.get(node.id, node.id), ctx=node.ctx), node)
        return node

def get_z_score(code):
    orig_score = get_score(code)
    p_scores = []
    for _ in range(NUM_PERTURBATIONS):
        try:
            tree = ast.parse(code)
            p_code = ast.unparse(AdvancedPerturber().visit(tree))
        except:
            p_code = code + (" " * random.randint(1, 5))
        p_scores.append(get_score(p_code))
    
    p_scores = np.array(p_scores)
    std = p_scores.std()
    return (orig_score - p_scores.mean()) / std if std > 0 else 0.0

if __name__ == "__main__":
    # 1. PHASE 1: Find Threshold using Train Set
    print("\n[1/2] Finding optimal threshold from train.parquet...")
    train_df = pd.read_parquet('data/train.parquet').head(TRAIN_SAMPLES)
    train_z = [get_z_score(c) for c in tqdm(train_df['code'])]
    train_labels = train_df['label'].tolist()

    best_f1, best_threshold = 0, 0
    # Constrain search to avoid the "predict all ones" trap
    search_range = np.linspace(np.percentile(train_z, 15), np.percentile(train_z, 85), 100)
    for t in search_range:
        f1 = f1_score(train_labels, [1 if z > t else 0 for z in train_z])
        if f1 > best_f1:
            best_f1, best_threshold = f1, t

    print(f"Best Threshold set to: {best_threshold:.4f} (Train F1: {best_f1:.4f})")

    # 2. PHASE 2: Predict on Test Set
    print("\n[2/2] Generating predictions for test.parquet...")
    test_df = pd.read_parquet('data/test.parquet')
    
    results = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        z = get_z_score(row['code'])
        label = 1 if z > best_threshold else 0
        results.append({
            "ID": row['ID'],
            "label": label
        })

    # 3. Save to CSV
    submission_df = pd.DataFrame(results)
    submission_df.to_csv('submission.csv', index=False)
    
    print("\n" + "="*30)
    print("DONE! Created 'submission.csv'")
    print("="*30)
    print(submission_df.head())