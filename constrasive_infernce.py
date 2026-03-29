import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


MODEL_NAME = "microsoft/graphcodebert-base"


class CodeGeneralizerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(MODEL_NAME)
        hidden_size = self.encoder.config.hidden_size

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attn_mask, mode='classification'):
        outputs = self.encoder(input_ids, attention_mask=attn_mask)
        pooled = outputs.last_hidden_state[:, 0, :]

        if mode == 'contrastive':
            z = self.projection_head(pooled)
            return F.normalize(z, dim=1)
        return self.classifier(pooled)


class TestCodeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.ids = dataframe["ID"].values
        self.code = dataframe["code"].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.code[idx]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "id": int(self.ids[idx]),
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using the contrastive-trained checkpoint.")
    parser.add_argument("--checkpoint", default="model_best_ood.pt", help="Path to trained checkpoint.")
    parser.add_argument("--test-path", default="data/test.parquet", help="Path to test parquet file.")
    parser.add_argument("--output-csv", default="submission_constrasive.csv", help="Path to output CSV.")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size.")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_df = pd.read_parquet(args.test_path)
    test_ds = TestCodeDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = CodeGeneralizerModel()
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            logits = model(ids, mask, mode="classification")
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            sample_ids = batch["id"].tolist()

            for sample_id, pred in zip(sample_ids, preds):
                rows.append({"ID": sample_id, "label": pred})

    submission_df = pd.DataFrame(rows, columns=["ID", "label"])
    submission_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(submission_df)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
