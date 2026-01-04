import argparse
import datetime
import json
import logging
import os
import random
import re
import sys
import importlib
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


DATASET_DIRS = [
    "dataset/wanglab___kegg",
    "dataset/wanglab___variant_effect_coding",
    "dataset/wanglab___variant_effect_non_snv",
]

LOGGER = logging.getLogger("llm_baselines")

ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)
NON_PATHOGENIC_RE = re.compile(r"\b(non[- ]?pathogenic|not pathogenic)\b")

LLM_PRESETS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}


def normalize_label(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text.strip())


def load_split(csv_path: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ["__index_level_0__"] if c in df.columns])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def get_split_paths(dataset_dir: str) -> Dict[str, str]:
    splits = {}
    for split_name in ["train", "validation", "val", "test"]:
        path = os.path.join(dataset_dir, f"{split_name}.csv")
        if os.path.exists(path):
            key = "validation" if split_name == "val" else split_name
            splits[key] = path
    return splits


def parse_labels(df: pd.DataFrame, dataset_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    if "cleaned_pathogenicity" in df.columns:
        labels = df["cleaned_pathogenicity"].astype(str).str.strip().str.lower().tolist()
        return df, labels

    if "answer" in df.columns:
        if "variant_effect" in dataset_dir:
            labels = (
                df["answer"]
                .astype(str)
                .str.split(";", n=1)
                .str[0]
                .str.strip()
                .str.lower()
                .tolist()
            )
            valid = {"pathogenic", "benign"}
            mask = [label in valid for label in labels]
            if not all(mask):
                df = df.loc[mask].reset_index(drop=True)
                labels = [label for label in labels if label in valid]
            return df, labels
        labels = df["answer"].astype(str).str.strip().str.lower().tolist()
        return df, labels

    raise ValueError(f"No label column found for {dataset_dir}")


def get_text_column(df: pd.DataFrame, text_column: Optional[str]) -> str:
    if text_column:
        if text_column not in df.columns:
            raise ValueError(f"Requested text column '{text_column}' not found.")
        return text_column
    if "question" in df.columns:
        return "question"
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("No text column found.")


def sample_few_shot(df: pd.DataFrame, labels: List[str], k: int, seed: int):
    if k <= 0:
        return []
    random.seed(seed)
    indices = list(range(len(df)))
    random.shuffle(indices)
    return indices[:k]


def build_kegg_prompt(
    label_list: List[str],
    few_shot: List[Tuple[str, str]],
) -> str:
    label_text = ", ".join(label_list)
    parts = [
        "You are a biomedical assistant. Choose exactly one disease label.",
        "Output format: Final answer: <DISEASE_NAME>",
        f"Labels: {label_text}",
    ]
    for example_text, example_label in few_shot:
        parts.append("Example:")
        parts.append(f"Input: {example_text}")
        parts.append(f"Final answer: {example_label}")
    parts.append("Input:")
    return "\n".join(parts)


def build_vep_prompt(few_shot: List[Tuple[str, str]]) -> str:
    parts = [
        "You are a biomedical assistant. Classify the variant as BENIGN or PATHOGENIC.",
        "Output format: Final answer: BENIGN or Final answer: PATHOGENIC",
    ]
    for example_text, example_label in few_shot:
        parts.append("Example:")
        parts.append(f"Input: {example_text}")
        parts.append(f"Final answer: {example_label.upper()}")
    parts.append("Input:")
    return "\n".join(parts)


def extract_prediction(text: str, label_map: Dict[str, str]) -> str:
    match = ANSWER_RE.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.splitlines()[0]
    normalized = normalize_label(candidate)

    if NON_PATHOGENIC_RE.search(normalized):
        return "benign"
    if "benign" in normalized and "pathogenic" not in normalized:
        return "benign"
    if "pathogenic" in normalized:
        return "pathogenic"

    for label_norm, label_raw in label_map.items():
        if normalized == label_norm:
            return label_raw

    for label_norm, label_raw in sorted(label_map.items(), key=lambda x: -len(x[0])):
        if label_norm in normalized:
            return label_raw

    return "unknown"


def get_label_list(train_labels: List[str]) -> List[str]:
    unique = sorted(set(train_labels))
    return unique


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def get_max_input_tokens(tokenizer, model, override: Optional[int]) -> int:
    if override:
        return override
    if hasattr(model.config, "max_position_embeddings"):
        return int(model.config.max_position_embeddings)
    if hasattr(model.config, "max_seq_len"):
        return int(model.config.max_seq_len)
    if hasattr(model.config, "max_seq_length"):
        return int(model.config.max_seq_length)
    if hasattr(tokenizer, "model_max_length"):
        if 0 < tokenizer.model_max_length < 100000:
            return int(tokenizer.model_max_length)
    return 2048


def truncate_question(
    prefix: str,
    question: str,
    tokenizer,
    max_prompt_tokens: int,
    safety_margin: int = 64,
) -> str:
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    question_ids = tokenizer(question, add_special_tokens=False).input_ids
    budget = max_prompt_tokens - len(prefix_ids) - safety_margin
    if budget < 0:
        budget = 0
    if len(question_ids) > budget:
        question_ids = question_ids[:budget]
        question = tokenizer.decode(question_ids, skip_special_tokens=True)
    return prefix + question


def run_dataset(
    dataset_dir: str,
    tokenizer,
    model,
    device: torch.device,
    max_new_tokens: int,
    max_input_tokens: int,
    use_chat_template: bool,
    split: str,
    max_samples: Optional[int],
    seed: int,
    few_shot_k: int,
    output_dir: str,
    preset: Optional[str],
    quantization: Optional[str],
    device_map: Optional[str],
    text_column: Optional[str],
) -> None:
    split_paths = get_split_paths(dataset_dir)
    if "train" not in split_paths or split not in split_paths:
        LOGGER.warning("Skipping %s: missing train/%s CSV.", dataset_dir, split)
        return

    train_df = load_split(split_paths["train"], None, seed)
    eval_df = load_split(split_paths[split], max_samples, seed)

    train_df, train_labels = parse_labels(train_df, dataset_dir)
    eval_df, eval_labels = parse_labels(eval_df, dataset_dir)

    text_col = get_text_column(eval_df, text_column)
    train_text_col = get_text_column(train_df, text_column)

    few_shot_indices = sample_few_shot(train_df, train_labels, few_shot_k, seed)
    few_shot = [
        (train_df.iloc[i][train_text_col], train_labels[i]) for i in few_shot_indices
    ]

    label_list = get_label_list(train_labels)
    label_map = {normalize_label(label): label for label in label_list}

    predictions = []
    raw_outputs = []

    if "kegg" in dataset_dir:
        prompt_prefix = build_kegg_prompt(label_list, few_shot)
    else:
        prompt_prefix = build_vep_prompt(few_shot)

    LOGGER.info("Running %s split for %s", split, dataset_dir)
    for text in tqdm(eval_df[text_col].astype(str).tolist(), desc="Generating"):
        prompt = truncate_question(
            prompt_prefix,
            text,
            tokenizer,
            max_prompt_tokens=max_input_tokens - max_new_tokens,
        )
        formatted = format_prompt(tokenizer, prompt, use_chat_template)
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        raw_outputs.append(decoded)
        predictions.append(extract_prediction(decoded, label_map))

    eval_labels_norm = [normalize_label(label) for label in eval_labels]
    pred_labels_norm = [normalize_label(label) for label in predictions]
    all_labels = sorted(set(eval_labels_norm) | set(pred_labels_norm))

    accuracy = accuracy_score(eval_labels_norm, pred_labels_norm)
    f1 = f1_score(eval_labels_norm, pred_labels_norm, labels=all_labels, average="macro")

    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    metrics_path = os.path.join(output_path, "metrics.json")

    metrics = {
        "split": split,
        "accuracy": float(accuracy),
        "f1": float(f1),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "metrics": metrics,
                "settings": {
                    "preset": preset,
                    "quantization": quantization,
                    "device_map": device_map,
                    "model": model.name_or_path,
                    "max_new_tokens": max_new_tokens,
                    "few_shot_k": few_shot_k,
                    "split": split,
                    "use_chat_template": use_chat_template,
                },
            },
            f,
            indent=2,
        )

    preds_df = pd.DataFrame(
        {
            "text": eval_df[text_col].astype(str),
            "label": eval_labels_norm,
            "prediction": pred_labels_norm,
            "raw_output": raw_outputs,
        }
    )
    preds_path = os.path.join(output_path, "predictions.csv")
    preds_df.to_csv(preds_path, index=False)

    LOGGER.info("Saved metrics to %s", metrics_path)
    LOGGER.info("Saved predictions to %s", preds_path)


def apply_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    model_name = LLM_PRESETS.get(args.preset)
    if not model_name:
        raise ValueError(f"Unknown preset: {args.preset}")
    args.model = model_name


def require_bitsandbytes() -> None:
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit/8-bit loading. "
            "Install with: pip install bitsandbytes"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-only baselines")
    parser.add_argument(
        "--preset",
        choices=sorted(LLM_PRESETS.keys()),
        help="Model preset (overrides --model).",
    )
    parser.add_argument(
        "--list_presets",
        action="store_true",
        help="List available model presets and exit.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASET_DIRS,
        help="Dataset directories containing train/test CSVs.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model name for the LLM.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--few_shot_k", type=int, default=3)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument(
        "--bnb_compute_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--device_map",
        default=None,
        help="Optional device map for model loading (e.g. auto).",
    )
    parser.add_argument("--output_dir", default="outputs/llm_baselines")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--text_column", default="question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        for name, model_name in LLM_PRESETS.items():
            print(f"{name}: {model_name}")
        sys.exit(0)
    apply_preset(args)
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Choose only one of --load_in_4bit or --load_in_8bit.")
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or timestamp
    run_output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(run_output_dir, exist_ok=True)

    log_path = os.path.join(run_output_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    LOGGER.info("Logging to %s", log_path)
    LOGGER.info("Run output dir: %s", run_output_dir)
    LOGGER.info("Args: %s", vars(args))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.unk_token_id is not None:
            tokenizer.pad_token_id = tokenizer.unk_token_id

    quantization = None
    quant_config = None
    device_map = args.device_map
    if args.load_in_4bit or args.load_in_8bit:
        require_bitsandbytes()
        if BitsAndBytesConfig is None:
            raise RuntimeError("BitsAndBytesConfig is unavailable in this environment.")
        quantization = "4bit" if args.load_in_4bit else "8bit"
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.bnb_compute_dtype]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if device_map is None:
            device_map = "auto"

    torch_dtype = "auto" if device.type == "cuda" else torch.float32
    if quant_config is not None:
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_config,
    )
    if device_map is None:
        model = model.to(device)
    model.eval()

    max_input_tokens = get_max_input_tokens(tokenizer, model, args.max_input_tokens)
    LOGGER.info("Max input tokens: %s", max_input_tokens)

    for dataset_dir in args.datasets:
        if not os.path.isdir(dataset_dir):
            LOGGER.warning("Missing dataset dir: %s", dataset_dir)
            continue
        run_dataset(
            dataset_dir=dataset_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=max_input_tokens,
            use_chat_template=args.use_chat_template,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            few_shot_k=args.few_shot_k,
            output_dir=run_output_dir,
            preset=args.preset,
            quantization=quantization,
            device_map=device_map,
            text_column=args.text_column,
        )


if __name__ == "__main__":
    main()
