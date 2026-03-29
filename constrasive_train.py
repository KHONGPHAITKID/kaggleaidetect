import numpy as np
import pandas as pd
import re
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ===== AUGMENTATION =====

def augment_code(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

    words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
    unique_words = list(set([w for w in words if len(w) > 3]))

    mapping = {word: f"var_{i}" for i, word in enumerate(unique_words)}

    for word, replacement in mapping.items():
        keywords = {'def', 'class', 'if', 'else', 'return'}
        if word not in keywords:
            code = re.sub(r'\b' + word + r'\b', replacement, code)

    return code


# ===== MODEL =====

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(1, torch.arange(batch_size, device=device).view(-1, 1), 0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
        return -mean_log_prob_pos.mean()


class CodeGeneralizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        hidden_size = self.encoder.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)
        )

        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attn_mask, mode='contrastive'):
        outputs = self.encoder(input_ids, attention_mask=attn_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        if mode == 'contrastive':
            z = self.projection_head(pooled)
            return F.normalize(z, dim=1)
        return self.classifier(pooled)


# ===== DATASET =====

class CodeSnippetDataset(Dataset):
    def __init__(self, dataframe, tokenizer, is_train=False):
        self.code = dataframe['code'].values
        self.labels = dataframe['label'].values
        self.tokenizer = tokenizer
        self.is_train = is_train # Add a flag to only augment training data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        snippet = str(self.code[idx])
        
        # APPLY AUGMENTATION HERE (Only for training data)
        if self.is_train and np.random.rand() > 0.5: # 50% chance to augment
            snippet = augment_code(snippet)

        encoding = self.tokenizer(
            snippet,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx]).long()
        }


def main():
    # ===== LOAD DATA =====
    df = pd.read_parquet('data/train.parquet')
    print(df.head())

    unique_languages = df['language'].unique()
    print(unique_languages)

    counts = df['language'].value_counts()

    print(f"Python: {counts.get('Python', 0)}")
    print(f"C++: {counts.get('C++', 0)}")
    print(f"Java: {counts.get('Java', 0)}")

    # ===== DATA PREPARATION =====
    df_train_old = pd.read_parquet('data/train.parquet')
    df_val_old = pd.read_parquet('data/validation.parquet')

    df_all = pd.concat([df_train_old, df_val_old], ignore_index=True)

    df_python = df_all[df_all['language'] == 'Python']
    df_cpp = df_all[df_all['language'] == 'C++']
    df_java = df_all[df_all['language'] == 'Java']

    target_python = min(50000, len(df_python))
    df_python_sampled = df_python.sample(n=target_python, random_state=42)

    new_train = pd.concat([df_cpp, df_python_sampled], ignore_index=True).sample(frac=1, random_state=42)
    new_validation = df_java.copy()

    # ===== TRAINING SETUP =====
    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")

    train_ds = CodeSnippetDataset(new_train, tokenizer, is_train=True)
    val_ds = CodeSnippetDataset(new_validation, tokenizer, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeGeneralizerModel().to(device)

    # # ===== CONTRASTIVE TRAINING =====
    # criterion = SupConLoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # for epoch in range(3):
    #     model.train()
    #     total_loss = 0

    #     loop = tqdm(train_loader, desc=f"Contrastive Epoch {epoch+1}")

    #     for batch in loop:
    #         ids = batch['input_ids'].to(device)
    #         mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)

    #         embeddings = model(ids, mask)
    #         loss = criterion(embeddings, labels)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
    #         loop.set_postfix(loss=loss.item())

    #     print(f"Epoch {epoch+1} avg loss: {total_loss / len(train_loader):.4f}")

    # # ===== CLASSIFICATION PHASE =====
    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()

    # best_f1 = float("-inf")
    # save_path = "model_best_ood.pt"

    # for epoch in range(5):
    #     model.train()
    #     model.encoder.eval()
    #     total_train_loss = 0

    #     train_loop = tqdm(train_loader, desc=f"Classification Train Epoch {epoch+1}")

    #     for batch in train_loop:
    #         ids = batch['input_ids'].to(device)
    #         mask = batch['attention_mask'].to(device)
    #         labels = batch['labels'].to(device)

    #         logits = model(ids, mask, mode='classification')
    #         loss = criterion(logits, labels)

    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_train_loss += loss.item()
    #         train_loop.set_postfix(loss=loss.item())

    #     avg_train_loss = total_train_loss / len(train_loader)

    #     model.eval()
    #     val_preds = []
    #     val_trues = []
    #     total_val_loss = 0

    #     val_loop = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}", leave=False)

    #     with torch.no_grad():
    #         for batch in val_loop:
    #             ids = batch['input_ids'].to(device)
    #             mask = batch['attention_mask'].to(device)
    #             labels = batch['labels'].to(device)

    #             logits = model(ids, mask, mode='classification')
    #             loss = criterion(logits, labels)
    #             total_val_loss += loss.item()

    #             preds = torch.argmax(logits, dim=1)
    #             val_preds.extend(preds.cpu().numpy())
    #             val_trues.extend(labels.cpu().numpy())

    #     avg_val_loss = total_val_loss / len(val_loader)

    #     acc = accuracy_score(val_trues, val_preds)
    #     prec = precision_score(val_trues, val_preds, average='macro', zero_division=0)
    #     rec = recall_score(val_trues, val_preds, average='macro', zero_division=0)
    #     f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)

    #     print(f"\nEpoch {epoch+1} Summary:")
    #     print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    #     print(f"Val Acc: {acc:.4f} | Val Precision: {prec:.4f} | Val Recall: {rec:.4f} | Val F1: {f1:.4f}")

    #     if f1 > best_f1:
    #         best_f1 = f1
    #         torch.save(model.state_dict(), save_path)
    #         print(f"New best F1 score! Model saved to {save_path}")
    #     print("-" * 50)

    # print(f"\nTraining Complete. Best Validation F1: {best_f1:.4f}")
    # ===== END-TO-END FINE TUNING =====
    # Ensure ALL parameters (encoder + classifier) are trainable
    for param in model.parameters():
        param.requires_grad = True

    # Use a small learning rate since we are updating the pre-trained encoder
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    best_f1 = float("-inf")
    save_path = "model_best_ood.pt"

    for epoch in range(3): # 3 epochs is usually plenty for full fine-tuning
        model.train()
        total_train_loss = 0

        train_loop = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")

        for batch in train_loop:
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # We pass mode='classification' directly
            logits = model(ids, mask, mode='classification')
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        # --- VALIDATION ---
        model.eval()
        val_preds, val_trues = [], []
        total_val_loss = 0

        val_loop = tqdm(val_loader, desc=f"Val Epoch {epoch+1}", leave=False)

        with torch.no_grad():
            for batch in val_loop:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(ids, mask, mode='classification')
                loss = criterion(logits, labels)
                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_trues.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        
        acc = accuracy_score(val_trues, val_preds)
        prec = precision_score(val_trues, val_preds, average='macro', zero_division=0)
        rec = recall_score(val_trues, val_preds, average='macro', zero_division=0)
        f1 = f1_score(val_trues, val_preds, average='macro', zero_division=0)

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val Acc: {acc:.4f} | Val Precision: {prec:.4f} | Val Recall: {rec:.4f} | Val F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New best F1 score! Model saved to {save_path}")
        print("-" * 50)


if __name__ == "__main__":
    main()
