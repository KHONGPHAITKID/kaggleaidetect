import pandas as pd
import torch
from torch.utils.data import Dataset, Subset
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments
)
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score


MODEL_NAME = "microsoft/codebert-base"
MAX_LEN = 512


class CodeDataset(Dataset):
    def __init__(self, df, tokenizer, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code = self.df.iloc[idx]["code"]

        enc = self.tokenizer(
            code,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            item["labels"] = torch.tensor(self.df.iloc[idx]["label"]).long()

        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds)
    }


class RandomEvalSubsetTrainer(Trainer):
    def __init__(self, *args, eval_sample_size=100, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_sample_size = eval_sample_size

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        if self.eval_sample_size is not None and len(eval_dataset) > self.eval_sample_size:
            indices = np.random.choice(len(eval_dataset), self.eval_sample_size, replace=False)
            eval_dataset = Subset(eval_dataset, indices.tolist())

        return super().get_eval_dataloader(eval_dataset)


def main():

    print("Loading datasets...")
    train_df = pd.read_parquet("data/train.parquet")
    val_df = pd.read_parquet("data/validation.parquet")
    test_df = pd.read_parquet("data/test.parquet")

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = CodeDataset(train_df, tokenizer)
    val_dataset = CodeDataset(val_df, tokenizer)
    test_dataset = CodeDataset(test_df, tokenizer, is_test=True)

    train_batch_size = 8
    steps_per_epoch = math.ceil(len(train_dataset) / train_batch_size)
    eval_steps = max(1, math.ceil(steps_per_epoch * 0.1))
    print(f"Evaluating every {eval_steps} steps (~0.1 epoch) on 100 random val samples.")

    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        use_safetensors=True
    )

    training_args = TrainingArguments(
        output_dir="./model",
        learning_rate=2e-5,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_dir="./logs",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none"
    )

    trainer = RandomEvalSubsetTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        eval_sample_size=100,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")
    trainer.train()

    print("Saving best model...")
    trainer.save_model("./best_codebert_model") 
    tokenizer.save_pretrained("./best_codebert_model")

    print("Evaluating...")
    trainer.evaluate()

    if True:
        print("Running inference on test set...")
        preds = trainer.predict(test_dataset)

        logits = preds.predictions
        labels = np.argmax(logits, axis=1)

        submission = pd.DataFrame({
            "ID": test_df["ID"],
            "label": labels
        })

        submission.to_csv("submission.csv", index=False)

        print("submission.csv saved!")


if __name__ == "__main__":
    main()
