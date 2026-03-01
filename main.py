import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dataset import get_score


MODEL_LIST = [
    # "google/gemma-2-9b-it",
    # "mistralai/Mistral-7B-Instruct-v0.3",
    # "Qwen/Qwen2.5-Coder-7B-Instruct",
    # "Nanbeige/Nanbeige4.1-3B",
    "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
]

PROMPT_TYPES = ("zero_shot", "one_shot", "few_shot")
DAGSHUB_OWNER = "nghessss"
DAGSHUB_REPO = "semivalA"
NON_QUANTIZED_MODELS = {
    "Nanbeige/Nanbeige4.1-3B",
    "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF",
}

EXAMPLE_HUMAN = (
    "import sys\n"
    "inf = float('inf')\n"
    "from bisect import bisect_left, bisect_right\n\n"
    "def get_array():\n"
    "    return list(map(int, sys.stdin.readline().strip().split()))\n\n"
    "n = int(input())\n"
    "arr = get_array()\n"
    "total = sum(arr)\n"
    "print(total)\n"
)

EXAMPLE_AI = (
    "def clamp(value):\n"
    "    return min(max(value, -10), 10)\n\n"
    "def dot(a, b):\n"
    "    return sum(x * y for x, y in zip(a, b))\n\n"
    "def main():\n"
    "    print(dot([1, 2, 3], [4, 5, 6]))\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)


def build_prompt_catalog():
    return {
        "google/gemma-2-9b-it": {
            "zero_shot": (
                "You are a strict code-origin classifier.\n"
                "Task: classify code as AI-generated (1) or human-written (0).\n"
                "Return exactly one character: 0 or 1.\n\n"
                "Code:\n```{code}```\nAnswer:"
            ),
            "one_shot": (
                "You are a strict code-origin classifier.\n"
                "Task: classify code as AI-generated (1) or human-written (0).\n"
                "Return exactly one character: 0 or 1.\n\n"
                "Example:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Code:\n```{code}```\nAnswer:"
            ),
            "few_shot": (
                "You are a strict code-origin classifier.\n"
                "Task: classify code as AI-generated (1) or human-written (0).\n"
                "Return exactly one character: 0 or 1.\n\n"
                "Example 1:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Example 2:\n"
                f"Code:\n```{EXAMPLE_AI}```\nLabel: 1\n\n"
                "Code:\n```{code}```\nAnswer:"
            ),
        },
        "mistralai/Mistral-7B-Instruct-v0.3": {
            "zero_shot": (
                "[INST] Decide if the code is AI-generated (1) or human-written (0). "
                "Output only 0 or 1.\n\nCode:\n```{code}``` [/INST]"
            ),
            "one_shot": (
                "[INST] Decide if the code is AI-generated (1) or human-written (0). "
                "Output only 0 or 1.\n\n"
                "Example:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Code:\n```{code}``` [/INST]"
            ),
            "few_shot": (
                "[INST] Decide if the code is AI-generated (1) or human-written (0). "
                "Output only 0 or 1.\n\n"
                "Example 1:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Example 2:\n"
                f"Code:\n```{EXAMPLE_AI}```\nLabel: 1\n\n"
                "Code:\n```{code}``` [/INST]"
            ),
        },
        "Qwen/Qwen2.5-Coder-7B-Instruct": {
            "zero_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\nCode:\n```{code}```\nLabel:"
            ),
            "one_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\n"
                "Example:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Code:\n```{code}```\nLabel:"
            ),
            "few_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\n"
                "Example 1:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Example 2:\n"
                f"Code:\n```{EXAMPLE_AI}```\nLabel: 1\n\n"
                "Code:\n```{code}```\nLabel:"
            ),
        },
        "TeichAI/Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill-GGUF": {
            "zero_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\nCode:\n```{code}```\nLabel:"
            ),
            "one_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\n"
                "Example:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Code:\n```{code}```\nLabel:"
            ),
            "few_shot": (
                "Classify this code as AI-generated (1) or human-written (0).\n"
                "Only output 0 or 1.\n\n"
                "Example 1:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nLabel: 0\n\n"
                "Example 2:\n"
                f"Code:\n```{EXAMPLE_AI}```\nLabel: 1\n\n"
                "Code:\n```{code}```\nLabel:"
            ),
        },
        "Nanbeige/Nanbeige4.1-3B": {
            "zero_shot": (
                "You are a binary classifier for source attribution in code.\n"
                "1 means AI-generated, 0 means human-written.\n"
                "Return exactly one token: 0 or 1.\n\n"
                "Input code:\n```{code}```\nPrediction:"
            ),
            "one_shot": (
                "You are a binary classifier for source attribution in code.\n"
                "1 means AI-generated, 0 means human-written.\n"
                "Return exactly one token: 0 or 1.\n\n"
                "Example:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nPrediction: 0\n\n"
                "Input code:\n```{code}```\nPrediction:"
            ),
            "few_shot": (
                "You are a binary classifier for source attribution in code.\n"
                "1 means AI-generated, 0 means human-written.\n"
                "Return exactly one token: 0 or 1.\n\n"
                "Example 1:\n"
                f"Code:\n```{EXAMPLE_HUMAN}```\nPrediction: 0\n\n"
                "Example 2:\n"
                f"Code:\n```{EXAMPLE_AI}```\nPrediction: 1\n\n"
                "Input code:\n```{code}```\nPrediction:"
            ),
        }
    }


def safe_model_name(model_id):
    return model_id.replace("/", "_").replace("-", "_")


def parse_binary_prediction(raw_text):
    text = raw_text.strip()
    if "1" in text and "0" in text:
        # If both appear, prefer first appearing token.
        idx0 = text.find("0")
        idx1 = text.find("1")
        return 1 if idx1 < idx0 else 0
    if "1" in text:
        return 1
    return 0


def build_predictor(model, tokenizer, prompt_template, max_new_tokens, temperature, device):
    def predict(code):
        prompt = prompt_template.format(code=code)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = output_tokens[0][inputs.input_ids.shape[-1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        return parse_binary_prediction(decoded)

    return predict


def log_confusion_matrix(prefix, conf_matrix):
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        mlflow.log_metric(f"{prefix}_tn", int(tn))
        mlflow.log_metric(f"{prefix}_fp", int(fp))
        mlflow.log_metric(f"{prefix}_fn", int(fn))
        mlflow.log_metric(f"{prefix}_tp", int(tp))


def round2(value):
    return round(float(value), 2)


def get_model_load_kwargs(model_id, bnb_config, hf_token):
    common = {
        "device_map": "auto",
        "token": hf_token or None,
    }
    if model_id in NON_QUANTIZED_MODELS:
        return {
            **common,
            "torch_dtype": torch.bfloat16,
        }
    return {
        **common,
        "quantization_config": bnb_config,
        "torch_dtype": torch.float16,
    }


def run_benchmark(args):
    if args.use_dagshub:
        try:
            import dagshub

            dagshub.init(
                repo_owner=args.dagshub_owner,
                repo_name=args.dagshub_repo,
                mlflow=True,
            )
            print(
                f"DagsHub tracking enabled for "
                f"{args.dagshub_owner}/{args.dagshub_repo}"
            )
        except ImportError as exc:
            raise RuntimeError(
                "DagsHub tracking requested but 'dagshub' is not installed. "
                "Install it with: pip install dagshub"
            ) from exc

    prompt_catalog = build_prompt_catalog()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    session_name = f"prompt_model_versioning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    mlflow.set_experiment(args.experiment_name)
    for model_id in tqdm(MODEL_LIST, desc="Models", position=0):
        print(f"\nLoading model: {model_id}")
        tokenizer = None
        model = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token or None)
            model_load_kwargs = get_model_load_kwargs(model_id, bnb_config, args.hf_token)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_load_kwargs,
            )
            model.eval()
        except Exception as exc:
            fail_run_name = f"{safe_model_name(model_id)}__load_failed"
            error_text = str(exc)
            print(f"Skipping model due to load error: {model_id}\n{error_text}")

            with mlflow.start_run(run_name=fail_run_name):
                mlflow.set_tag("session_type", "multi_model_prompt_eval")
                mlflow.set_tag("session_name", session_name)
                mlflow.set_tag("run_status", "load_failed")
                mlflow.log_param("model_id", model_id)
                mlflow.log_param("error_type", type(exc).__name__)
                mlflow.log_param(
                    "quantization",
                    "none" if model_id in NON_QUANTIZED_MODELS else "4bit_nf4",
                )
                mlflow.log_text(error_text, f"errors/{fail_run_name}.txt")
            continue

        for prompt_type in tqdm(PROMPT_TYPES, desc=f"Prompts ({safe_model_name(model_id)})", position=1, leave=False):
            prompt_template = prompt_catalog[model_id][prompt_type]
            prompt_version = "v1"
            run_name = f"{safe_model_name(model_id)}__{prompt_type}__{prompt_version}"

            print(f"Evaluating: model={model_id}, prompt={prompt_type}")
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tag("session_type", "multi_model_prompt_eval")
                mlflow.set_tag("session_name", session_name)
                mlflow.log_param("n_samples", args.n_samples if args.n_samples else "all")
                mlflow.log_param("max_new_tokens", args.max_new_tokens)
                mlflow.log_param("temperature", args.temperature)
                mlflow.log_param("datasets", "train,validation")

                mlflow.log_param("model_id", model_id)
                mlflow.log_param("prompt_type", prompt_type)
                mlflow.log_param("prompt_version", prompt_version)
                if model_id in NON_QUANTIZED_MODELS:
                    mlflow.log_param("quantization", "none")
                    mlflow.log_param("compute_dtype", "bfloat16")
                else:
                    mlflow.log_param("quantization", "4bit_nf4")
                    mlflow.log_param("compute_dtype", "float16")

                mlflow.set_tag("model_name_safe", safe_model_name(model_id))
                mlflow.set_tag("prompt_id", f"{safe_model_name(model_id)}_{prompt_type}_{prompt_version}")
                prompt_preview = " ".join(prompt_template.split())[:180]
                mlflow.set_tag("prompt_preview", prompt_preview)

                predictor = build_predictor(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_template=prompt_template,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device,
                )

                train_metrics = get_score(
                    dataset="train",
                    get_predictions=predictor,
                    n_samples=args.n_samples,
                )
                val_metrics = get_score(
                    dataset="validation",
                    get_predictions=predictor,
                    n_samples=args.n_samples,
                )

                train_accuracy = round2(train_metrics["accuracy"])
                train_recall = round2(train_metrics["recall"])
                train_f1 = round2(train_metrics["f1_score"])
                val_accuracy = round2(val_metrics["accuracy"])
                val_recall = round2(val_metrics["recall"])
                val_f1 = round2(val_metrics["f1_score"])

                mlflow.log_metric("train_accuracy", train_accuracy)
                mlflow.log_metric("train_recall", train_recall)
                mlflow.log_metric("train_f1_macro", train_f1)
                log_confusion_matrix("train", np.array(train_metrics["confusion_matrix"]))

                mlflow.log_metric("val_accuracy", val_accuracy)
                mlflow.log_metric("val_recall", val_recall)
                mlflow.log_metric("val_f1_macro", val_f1)
                log_confusion_matrix("val", np.array(val_metrics["confusion_matrix"]))

                artifact_dir = Path("artifacts")
                artifact_dir.mkdir(exist_ok=True)
                prompt_path = artifact_dir / f"{run_name}_prompt.txt"
                metrics_path = artifact_dir / f"{run_name}_metrics.json"
                prompt_path.write_text(prompt_template, encoding="utf-8")
                metrics_path.write_text(
                    json.dumps(
                        {
                            "train": {
                                "accuracy": train_accuracy,
                                "recall": train_recall,
                                "f1_score": train_f1,
                                "confusion_matrix": np.array(train_metrics["confusion_matrix"]).tolist(),
                            },
                            "validation": {
                                "accuracy": val_accuracy,
                                "recall": val_recall,
                                "f1_score": val_f1,
                                "confusion_matrix": np.array(val_metrics["confusion_matrix"]).tolist(),
                            },
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(prompt_path), artifact_path="prompts")
                mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
                mlflow.log_text(prompt_template, f"prompts/{run_name}_prompt_inline.txt")

        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="MLflow benchmark for model+prompt versioning.")
    parser.add_argument("--experiment-name", type=str, default="ai_detect_prompt_versioning")
    parser.add_argument("--n-samples", type=int, default=None, help="Use subset of train/validation.")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--use-dagshub",
        action="store_true",
        help="Send MLflow tracking to DagsHub.",
    )
    parser.add_argument(
        "--dagshub-owner",
        type=str,
        default=DAGSHUB_OWNER,
        help="DagsHub repo owner.",
    )
    parser.add_argument(
        "--dagshub-repo",
        type=str,
        default=DAGSHUB_REPO,
        help="DagsHub repo name.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="HF token for gated/private models; reads HF_TOKEN by default.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
