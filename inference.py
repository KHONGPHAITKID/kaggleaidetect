import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROMPT_TYPES = ("zero_shot", "one_shot", "few_shot")
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
        },
    }


def parse_binary_prediction(raw_text):
    text = raw_text.strip()
    if "1" in text and "0" in text:
        idx0 = text.find("0")
        idx1 = text.find("1")
        return 1 if idx1 < idx0 else 0
    if "1" in text:
        return 1
    return 0


def get_model_load_kwargs(model_id, hf_token, use_quantized):
    common = {
        "device_map": "auto",
        "token": hf_token or None,
    }
    if use_quantized:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return {
            **common,
            "quantization_config": bnb_config,
            "dtype": torch.float16,
        }
    return {
        **common,
        "torch_dtype": torch.bfloat16,
    }


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


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for test set CSV generation.")
    parser.add_argument("--model-id", required=True, help="HuggingFace model id to run.")
    parser.add_argument(
        "--prompt-type",
        required=True,
        choices=PROMPT_TYPES,
        help="Prompting method: zero_shot, one_shot, or few_shot.",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Optional subset of test set.")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--flush-every", type=int, default=500, help="Append to CSV every N new rows.")
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN", ""))
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", help="Resume from existing output CSV.")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Start fresh and overwrite output CSV.")
    parser.set_defaults(resume=True)
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--quantized", action="store_true", help="Force 4-bit quantized loading.")
    quant.add_argument("--no-quantized", action="store_true", help="Force non-quantized loading.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompt_catalog = build_prompt_catalog()

    if args.model_id not in prompt_catalog:
        supported = ", ".join(sorted(prompt_catalog.keys()))
        raise ValueError(f"Model prompt not configured for '{args.model_id}'. Supported: {supported}")

    prompt_template = prompt_catalog[args.model_id][args.prompt_type]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.quantized:
        use_quantized = True
    elif args.no_quantized:
        use_quantized = False
    else:
        use_quantized = args.model_id not in NON_QUANTIZED_MODELS

    print(f"Loading model: {args.model_id}")
    print(f"Prompt type: {args.prompt_type}")
    print(f"Quantized: {use_quantized}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=args.hf_token or None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        **get_model_load_kwargs(args.model_id, args.hf_token, use_quantized),
    )
    model.eval()

    predictor = build_predictor(
        model=model,
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        device=device,
    )

    test_df = pd.read_parquet("data/test.parquet")
    if args.n_samples:
        test_df = test_df.head(args.n_samples)

    output_path = Path(args.output_csv)
    completed_ids = set()
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path, usecols=["ID"])
        completed_ids = set(existing_df["ID"].tolist())
        print(f"Resume enabled: found {len(completed_ids)} completed rows in {args.output_csv}")
    elif not args.resume and output_path.exists():
        output_path.unlink()
        print(f"Removed existing output file: {args.output_csv}")

    if completed_ids:
        remaining_df = test_df[~test_df["ID"].isin(completed_ids)]
    else:
        remaining_df = test_df

    total_rows = len(test_df)
    remaining_rows = len(remaining_df)
    print(f"Total rows: {total_rows} | Remaining rows to predict: {remaining_rows}")

    if remaining_rows == 0:
        print(f"Nothing to do. Output already complete: {args.output_csv}")
        return

    pending_rows = []
    wrote_any = output_path.exists()
    for _, row in tqdm(remaining_df.iterrows(), total=remaining_rows, desc="Predicting test rows"):
        pred = predictor(row["code"])
        pending_rows.append({"ID": row["ID"], "label": pred})

        if len(pending_rows) >= args.flush_every:
            chunk_df = pd.DataFrame(pending_rows)
            chunk_df.to_csv(
                output_path,
                mode="a",
                index=False,
                header=not wrote_any,
            )
            wrote_any = True
            pending_rows = []

    if pending_rows:
        chunk_df = pd.DataFrame(pending_rows)
        chunk_df.to_csv(
            output_path,
            mode="a",
            index=False,
            header=not wrote_any,
        )

    print(f"Saved predictions (resumable): {args.output_csv}")


if __name__ == "__main__":
    main()
