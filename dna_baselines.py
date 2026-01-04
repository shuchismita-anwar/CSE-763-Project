import argparse
import datetime
import inspect
import json
import logging
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
)


DATASET_DIRS = [
    "dataset/wanglab___kegg",
    "dataset/wanglab___variant_effect_coding",
    "dataset/wanglab___variant_effect_non_snv",
]

LABEL_CANDIDATES = [
    "cleaned_pathogenicity",
    "label",
    "labels",
    "answer",
    "target",
    "class",
]

ALT_SEQUENCE_CANDIDATES = [
    "variant_sequence",
    "mutated_sequence",
    "alt_sequence",
]

VALID_BASES_PATTERN = re.compile(r"[^ACGTN]")

LOGGER = logging.getLogger("dna_baselines")

MODEL_PRESETS = {
    "dnabert2": {
        "model": "zhihan1996/DNABERT-2-117M",
        "trust_remote_code": True,
    },
    "nt-v2-100m": {
        "model": "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        "trust_remote_code": True,
    },
    "nt-v2-500m": {
        "model": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
        "trust_remote_code": True,
    },
    "hyenadna-tiny-1k": {
        "model": "LongSafari/hyenadna-tiny-1k-seqlen-hf",
        "trust_remote_code": True,
    },
    "hyenadna-tiny-1k-d256": {
        "model": "LongSafari/hyenadna-tiny-1k-seqlen-d256-hf",
        "trust_remote_code": True,
    },
    "hyenadna-tiny-16k": {
        "model": "LongSafari/hyenadna-tiny-16k-seqlen-d128-hf",
        "trust_remote_code": True,
    },
    "hyenadna-small-32k": {
        "model": "LongSafari/hyenadna-small-32k-seqlen-hf",
        "trust_remote_code": True,
    },
    "hyenadna-medium-160k": {
        "model": "LongSafari/hyenadna-medium-160k-seqlen-hf",
        "trust_remote_code": True,
    },
    "hyenadna-medium-450k": {
        "model": "LongSafari/hyenadna-medium-450k-seqlen-hf",
        "trust_remote_code": True,
    },
    "hyenadna-large-1m": {
        "model": "LongSafari/hyenadna-large-1m-seqlen-hf",
        "trust_remote_code": True,
    },
}


def normalize_sequence(seq: str) -> str:
    seq = seq.upper()
    return VALID_BASES_PATTERN.sub("N", seq)


def find_label_column(columns: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def find_alt_column(columns: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in ALT_SEQUENCE_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def parse_labels(df: pd.DataFrame, dataset_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    label_col = find_label_column(list(df.columns))
    if label_col is None:
        raise ValueError(f"No label column found in {dataset_dir}")

    if label_col == "cleaned_pathogenicity":
        labels = df[label_col].astype(str).str.strip().str.lower().tolist()
        return df, labels

    if label_col == "answer":
        if "variant_effect" in dataset_dir:
            labels = (
                df[label_col]
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

        labels = df[label_col].astype(str).str.strip().str.lower().tolist()
        return df, labels

    labels = df[label_col].astype(str).str.strip().str.lower().tolist()
    return df, labels


def slice_window(seq: str, center: int, window_bp: int) -> str:
    if window_bp <= 0:
        return seq
    half = window_bp // 2
    start = max(0, center - half)
    end = start + window_bp
    if end > len(seq):
        end = len(seq)
        start = max(0, end - window_bp)
    return seq[start:end]


def find_variant_center(ref: str, alt: str) -> int:
    if not ref or not alt:
        return 0
    if len(ref) == len(alt):
        diffs = [i for i, (a, b) in enumerate(zip(ref, alt)) if a != b]
        if diffs:
            return (diffs[0] + diffs[-1]) // 2
        return len(ref) // 2

    min_len = min(len(ref), len(alt))
    start = None
    for i in range(min_len):
        if ref[i] != alt[i]:
            start = i
            break
    if start is None:
        start = min_len

    end_from_end = None
    ref_rev = ref[::-1]
    alt_rev = alt[::-1]
    for i in range(min_len):
        if ref_rev[i] != alt_rev[i]:
            end_from_end = i
            break
    if end_from_end is None:
        end_from_end = 0
    end = max(len(ref) - end_from_end - 1, start)
    return (start + end) // 2


def prepare_sequences(
    df: pd.DataFrame, window_bp: int
) -> Tuple[List[str], List[str]]:
    ref_col = "reference_sequence"
    if ref_col not in df.columns:
        raise ValueError("reference_sequence column missing.")
    alt_col = find_alt_column(list(df.columns))
    if alt_col is None:
        raise ValueError("No variant/mutated sequence column found.")

    ref_sequences = []
    alt_sequences = []
    for ref, alt in zip(df[ref_col].astype(str), df[alt_col].astype(str)):
        ref = normalize_sequence(ref)
        alt = normalize_sequence(alt)
        center = find_variant_center(ref, alt)
        ref_sequences.append(slice_window(ref, center, window_bp))
        alt_sequences.append(slice_window(alt, center, window_bp))
    return ref_sequences, alt_sequences


def get_split_paths(dataset_dir: str) -> Dict[str, str]:
    splits = {}
    for split_name in ["train", "validation", "val", "test"]:
        path = os.path.join(dataset_dir, f"{split_name}.csv")
        if os.path.exists(path):
            key = "validation" if split_name == "val" else split_name
            splits[key] = path
    return splits


def get_max_length(tokenizer, model, override: Optional[int]) -> int:
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
    return 512


def _filter_inputs_for_model(model, inputs):
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return inputs, True, True

    params = signature.parameters
    has_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    if has_kwargs:
        return inputs, "output_hidden_states" in params, "return_dict" in params

    allowed = set(params.keys())
    filtered = {key: value for key, value in inputs.items() if key in allowed}
    return filtered, "output_hidden_states" in allowed, "return_dict" in allowed


def _forward_model(model, inputs):
    model_inputs, supports_hidden, supports_return = _filter_inputs_for_model(
        model, inputs
    )
    kwargs = {}
    if supports_hidden:
        kwargs["output_hidden_states"] = True
    if supports_return:
        kwargs["return_dict"] = True
    try:
        return model(**model_inputs, **kwargs)
    except TypeError:
        return model(**model_inputs)


def _extract_last_hidden(outputs):
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states:
        return outputs.hidden_states[-1]
    if isinstance(outputs, tuple) and len(outputs) > 0:
        return outputs[0]
    return None


def forward_hidden_states(model, inputs):
    outputs = _forward_model(model, inputs)
    last_hidden = _extract_last_hidden(outputs)
    if last_hidden is not None:
        return last_hidden

    for attr in ["base_model", "esm", "model", "backbone"]:
        sub = getattr(model, attr, None)
        if sub is None or sub is model:
            continue
        outputs = _forward_model(sub, inputs)
        last_hidden = _extract_last_hidden(outputs)
        if last_hidden is not None:
            return last_hidden

    raise ValueError("Unable to extract hidden states from model outputs.")


def encode_sequences(
    sequences: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
) -> np.ndarray:
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for idx in tqdm(range(0, len(sequences), batch_size), desc="Embedding"):
            batch = sequences[idx : idx + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            if "attention_mask" not in inputs and "input_ids" in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            last_hidden = forward_hidden_states(model, inputs)
            if pooling == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                mask = inputs["attention_mask"].unsqueeze(-1)
                masked = last_hidden * mask
                pooled = masked.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_embeddings.append(pooled.cpu().numpy())
    return np.vstack(all_embeddings)


def combine_features(
    ref_emb: np.ndarray, alt_emb: np.ndarray, strategy: str
) -> np.ndarray:
    if strategy == "ref_only":
        return ref_emb
    if strategy == "alt_only":
        return alt_emb
    if strategy == "diff":
        return alt_emb - ref_emb
    if strategy == "concat_diff":
        return np.concatenate([ref_emb, alt_emb, alt_emb - ref_emb], axis=1)
    return np.concatenate([ref_emb, alt_emb], axis=1)


def encode_pairs(
    ref_sequences: List[str],
    alt_sequences: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
    strategy: str,
) -> np.ndarray:
    ref_emb = encode_sequences(
        ref_sequences, tokenizer, model, device, batch_size, max_length, pooling
    )
    alt_emb = encode_sequences(
        alt_sequences, tokenizer, model, device, batch_size, max_length, pooling
    )
    return combine_features(ref_emb, alt_emb, strategy)


def load_split(csv_path: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ["__index_level_0__"] if c in df.columns])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def train_classifier(x_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    classes = np.unique(y_train)
    if len(classes) > 2:
        solver = "lbfgs"
        multi_class = "multinomial"
    else:
        solver = "liblinear"
        multi_class = "auto"
    try:
        clf = LogisticRegression(
            max_iter=2000,
            solver=solver,
            multi_class=multi_class,
            class_weight="balanced",
            n_jobs=1,
        )
    except TypeError:
        # Older scikit-learn versions do not accept multi_class.
        clf = LogisticRegression(
            max_iter=2000,
            solver=solver,
            class_weight="balanced",
            n_jobs=1,
        )
    clf.fit(x_train, y_train)
    return clf


def evaluate(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, float]:
    if num_classes > 2:
        f1_avg = "macro"
    else:
        f1_avg = "binary"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=f1_avg)),
    }


def run_dataset(
    dataset_dir: str,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    pooling: str,
    strategy: str,
    window_bp: int,
    output_dir: str,
    max_samples: Optional[int],
    seed: int,
    preset: Optional[str],
) -> None:
    split_paths = get_split_paths(dataset_dir)
    if "train" not in split_paths or "test" not in split_paths:
        LOGGER.warning("Skipping %s: missing train/test CSV.", dataset_dir)
        return

    train_df = load_split(split_paths["train"], max_samples, seed)
    test_df = load_split(split_paths["test"], max_samples, seed)
    val_df = None
    if "validation" in split_paths:
        val_df = load_split(split_paths["validation"], max_samples, seed)

    train_df, train_labels = parse_labels(train_df, dataset_dir)
    test_df, test_labels = parse_labels(test_df, dataset_dir)
    if val_df is not None:
        val_df, val_labels = parse_labels(val_df, dataset_dir)
    else:
        val_labels = None

    ref_train, alt_train = prepare_sequences(train_df, window_bp)
    ref_test, alt_test = prepare_sequences(test_df, window_bp)
    if val_df is not None:
        ref_val, alt_val = prepare_sequences(val_df, window_bp)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_labels)
    y_test = label_encoder.transform(test_labels)
    y_val = label_encoder.transform(val_labels) if val_labels is not None else None

    LOGGER.info("Encoding train split for %s", dataset_dir)
    x_train = encode_pairs(
        ref_train,
        alt_train,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        pooling,
        strategy,
    )
    LOGGER.info("Encoding test split for %s", dataset_dir)
    x_test = encode_pairs(
        ref_test,
        alt_test,
        tokenizer,
        model,
        device,
        batch_size,
        max_length,
        pooling,
        strategy,
    )
    x_val = None
    if val_df is not None:
        LOGGER.info("Encoding validation split for %s", dataset_dir)
        x_val = encode_pairs(
            ref_val,
            alt_val,
            tokenizer,
            model,
            device,
            batch_size,
            max_length,
            pooling,
            strategy,
        )

    LOGGER.info("Training classifier for %s", dataset_dir)
    clf = train_classifier(x_train, y_train)
    test_pred = clf.predict(x_test)
    metrics = {"test": evaluate(y_test, test_pred, len(label_encoder.classes_))}
    if x_val is not None and y_val is not None:
        val_pred = clf.predict(x_val)
        metrics["validation"] = evaluate(y_val, val_pred, len(label_encoder.classes_))

    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    output_path = os.path.join(output_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(output_path, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "classifier": clf,
                "label_encoder": label_encoder,
                "settings": {
                    "preset": preset,
                    "model": model.name_or_path,
                    "window_bp": window_bp,
                    "pooling": pooling,
                    "strategy": strategy,
                    "max_length": max_length,
                    "batch_size": batch_size,
                },
            },
            f,
        )
    LOGGER.info("Saved model to %s", model_path)

    metrics_path = os.path.join(output_path, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": dataset_name,
                "num_classes": len(label_encoder.classes_),
                "classes": label_encoder.classes_.tolist(),
                "metrics": metrics,
                "settings": {
                    "preset": preset,
                    "model": model.name_or_path,
                    "window_bp": window_bp,
                    "pooling": pooling,
                    "strategy": strategy,
                    "max_length": max_length,
                    "batch_size": batch_size,
                },
            },
            f,
            indent=2,
        )
    LOGGER.info("Saved metrics to %s", metrics_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DNA-only baselines")
    parser.add_argument(
        "--preset",
        choices=sorted(MODEL_PRESETS.keys()),
        help="Model preset (overrides --model and --trust_remote_code).",
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
        default="zhihan1996/DNABERT-2-117M",
        help="HF model name for DNA encoder.",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--window_bp", type=int, default=512)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--strategy",
        choices=["concat", "diff", "concat_diff", "ref_only", "alt_only"],
        default="concat",
    )
    parser.add_argument("--output_dir", default="outputs/dna_baselines")
    parser.add_argument(
        "--run_name",
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def apply_model_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    preset = MODEL_PRESETS.get(args.preset)
    if preset is None:
        raise ValueError(f"Unknown preset: {args.preset}")
    args.model = preset["model"]
    if preset.get("trust_remote_code"):
        args.trust_remote_code = True


def main() -> None:
    args = parse_args()
    if args.list_presets:
        for name, preset in MODEL_PRESETS.items():
            print(f"{name}: {preset['model']}")
        sys.exit(0)
    apply_model_preset(args)
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
    if args.preset:
        LOGGER.info("Preset %s -> %s", args.preset, args.model)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    model = None
    try:
        model = AutoModel.from_pretrained(
            args.model, trust_remote_code=args.trust_remote_code
        )
    except ValueError as exc:
        msg = str(exc)
        if "config_class" in msg and args.trust_remote_code:
            LOGGER.warning(
                "Config class mismatch detected; retrying with BertConfig (no remote code)."
            )
            config = BertConfig.from_pretrained(args.model)
            model = AutoModel.from_pretrained(
                args.model, trust_remote_code=False, config=config
            )
        elif "Unrecognized configuration class" in msg and "AutoModel" in msg:
            LOGGER.warning(
                "AutoModel unsupported; trying AutoModelForMaskedLM then AutoModelForCausalLM."
            )
            try:
                model = AutoModelForMaskedLM.from_pretrained(
                    args.model, trust_remote_code=args.trust_remote_code
                )
            except Exception as inner_exc:
                LOGGER.warning("AutoModelForMaskedLM failed: %s", inner_exc)
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, trust_remote_code=args.trust_remote_code
                )
        else:
            raise
    if model is None:
        raise ValueError("Failed to load model.")
    model = model.to(device)

    max_length = get_max_length(tokenizer, model, args.max_length)
    for dataset_dir in args.datasets:
        if not os.path.isdir(dataset_dir):
            LOGGER.warning("Missing dataset dir: %s", dataset_dir)
            continue
        LOGGER.info("Running dataset: %s", dataset_dir)
        run_dataset(
            dataset_dir=dataset_dir,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=max_length,
            pooling=args.pooling,
            strategy=args.strategy,
            window_bp=args.window_bp,
            output_dir=run_output_dir,
            max_samples=args.max_samples,
            seed=args.seed,
            preset=args.preset,
        )


if __name__ == "__main__":
    main()
