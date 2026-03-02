# KaggleAIDeTect

This repo has two main workflows:

1. `main.py`: benchmark multiple models and prompt methods (`zero_shot`, `one_shot`, `few_shot`) on `train` + `validation`, tracked with MLflow (optionally DagsHub).
2. `inference.py`: run one chosen model + one chosen prompt method on `test.parquet` and export CSV predictions.

## 1) Setup

### Python environment

Use the same Python environment for install + run.

```powershell
python -m pip install -U pip
python -m pip install torch transformers bitsandbytes pandas pyarrow scikit-learn tqdm mlflow accelerate huggingface_hub
```

Optional (for DagsHub MLflow tracking):

```powershell
python -m pip install dagshub
```

### HF token

Set your token in `.env`:

```env
HF_TOKEN="your_hf_token_here"
```

Or set environment variable directly in terminal:

```powershell
$env:HF_TOKEN="your_hf_token_here"
```

## 2) Data

Expected files:

- `data/train.parquet`
- `data/validation.parquet`
- `data/test.parquet`

`dataset.py` is used for scoring and test CSV schema (`ID,label`).

## 3) Run Benchmark (`main.py`)

Runs model/prompt combinations on **train + validation** and logs to MLflow.

### Basic

```powershell
python main.py --experiment-name ai_detect_prompt_versioning --n-samples 100
```

### Full data

```powershell
python main.py --experiment-name ai_detect_prompt_versioning
```

### With DagsHub tracking

```powershell
python main.py --use-dagshub --dagshub-owner nghessss --dagshub-repo semivalA --experiment-name ai_detect_prompt_versioning
```

### Key options

- `--experiment-name`: MLflow experiment name.
- `--n-samples`: use subset of train/validation for quick testing.
- `--max-new-tokens`: generation output length (default `2`).
- `--temperature`: generation temperature (default `0.0`).
- `--hf-token`: override `HF_TOKEN` env value.
- `--use-dagshub`, `--dagshub-owner`, `--dagshub-repo`: DagsHub MLflow routing.

## 4) Run Inference (`inference.py`)

Runs one model + one prompt method on test set and outputs CSV.

### Example: Gemma + one-shot

```powershell
python inference.py --model-id "google/gemma-2-9b-it" --prompt-type one_shot --output-csv submission_gemma_oneshot.csv
```

### Resumable large test run (recommended for 500k rows)

```powershell
python inference.py --model-id "google/gemma-2-9b-it" --prompt-type one_shot --output-csv submission_gemma_oneshot.csv --resume --flush-every 1000
```

### Start fresh (overwrite existing output)

```powershell
python inference.py --model-id "google/gemma-2-9b-it" --prompt-type one_shot --output-csv submission_gemma_oneshot.csv --no-resume
```

### Key options

- `--model-id`: model to run (must be configured in prompt catalog in `inference.py`).
- `--prompt-type`: one of `zero_shot`, `one_shot`, `few_shot`.
- `--output-csv`: output file path (`ID,label`).
- `--n-samples`: optional subset of test rows for smoke test.
- `--resume` / `--no-resume`: continue from existing output or restart.
- `--flush-every`: append frequency to CSV for long runs.
- `--quantized` / `--no-quantized`: force load mode.
- `--hf-token`: override `HF_TOKEN` env value.

## 5) Notes on Model Memory

- Loading can fail before inference if CPU RAM / VRAM is insufficient.
- If you see memory errors:
  - use smaller model,
  - switch to quantized mode (`--quantized`) when possible,
  - close other memory-heavy apps,
  - increase Windows page file,
  - reduce parallel workloads.

## 6) Common Errors

- `requires accelerate`:
  - install in the same interpreter used to run scripts:
  ```powershell
  python -m pip install -U accelerate
  ```

- `DefaultCPUAllocator: not enough memory`:
  - model loading exceeded available RAM/pagefile.
  - this is not caused by resume/CSV logic.

