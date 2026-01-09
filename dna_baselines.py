import argparse
import datetime
import hashlib
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
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def parse_window_bps(window_bps: Optional[str], window_bp: int) -> List[int]:
    if window_bps:
        values = [int(val.strip()) for val in window_bps.split(",") if val.strip()]
        if not values:
            raise ValueError("window_bps must include at least one integer value.")
        return values
    return [window_bp]


def parse_hidden_sizes(hidden_sizes: str) -> Tuple[int, ...]:
    sizes = [int(val.strip()) for val in hidden_sizes.split(",") if val.strip()]
    if not sizes:
        raise ValueError("mlp_hidden must include at least one integer value.")
    return tuple(sizes)


def sanitize_slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    if len(slug) > max_len:
        digest = hashlib.md5(slug.encode("utf-8")).hexdigest()[:8]
        slug = f"{slug[: max_len - 9]}_{digest}"
    return slug


def short_model_tag(model_name: str) -> str:
    if not model_name:
        return "model"
    return model_name.split("/")[-1]


def build_approach_tag(
    preset: Optional[str],
    model_name: str,
    window_bps: List[int],
    pooling: str,
    strategy: str,
    add_variant_features: bool,
    add_pll_delta: bool,
    classifier: str,
    mlp_hidden: Tuple[int, ...],
    class_weight: Optional[str],
    standardize: bool,
) -> str:
    model_tag = preset or short_model_tag(model_name)
    window_tag = "w" + "-".join(str(val) for val in window_bps)
    parts = [
        model_tag,
        window_tag,
        f"pool-{pooling}",
        f"strat-{strategy}",
        f"clf-{classifier}",
    ]
    if classifier == "mlp" and mlp_hidden:
        parts.append("mlp-" + "x".join(str(val) for val in mlp_hidden))
    if add_variant_features:
        parts.append("varfeat")
    if add_pll_delta:
        parts.append("pll")
    if standardize:
        parts.append("std")
    if class_weight is not None:
        parts.append("cw")
    return sanitize_slug("__".join(parts))


def add_log_handler(log_path: str):
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    return handler


def compute_window_bounds(center: int, seq_len: int, window_bp: int) -> Tuple[int, int]:
    if window_bp <= 0 or seq_len == 0:
        return 0, max(seq_len - 1, 0)
    half = window_bp // 2
    start = max(0, center - half)
    end = start + window_bp - 1
    if end >= seq_len:
        end = seq_len - 1
        start = max(0, end - window_bp + 1)
    return start, end


def find_variant_span(ref: str, alt: str) -> Tuple[int, int, int, int]:
    if not ref or not alt:
        return 0, 0, 0, len(alt) - len(ref)
    min_len = min(len(ref), len(alt))
    diffs = [i for i in range(min_len) if ref[i] != alt[i]]
    if diffs:
        start = diffs[0]
        end = diffs[-1]
        mismatch_count = len(diffs)
    else:
        start = min_len
        end = min_len
        mismatch_count = 0
    len_delta = len(alt) - len(ref)
    span_len = max(1, mismatch_count, abs(len_delta))
    end = max(end, start + span_len - 1)
    return start, end, mismatch_count, len_delta


def compute_variant_info(ref: str, alt: str) -> Dict[str, float]:
    start, end, mismatch_count, len_delta = find_variant_span(ref, alt)
    center = find_variant_center(ref, alt)
    span_len = max(1, end - start + 1)
    if len_delta == 0 and mismatch_count == 1:
        variant_type = "snv"
    elif len_delta > 0:
        variant_type = "ins"
    elif len_delta < 0:
        variant_type = "del"
    else:
        variant_type = "complex"
    return {
        "start": float(start),
        "end": float(end),
        "span_len": float(span_len),
        "mismatch_count": float(mismatch_count),
        "len_delta": float(len_delta),
        "center": float(center),
        "ref_len": float(len(ref)),
        "variant_type": variant_type,
    }


def collect_variant_infos(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[Dict[str, float]]]:
    ref_col = "reference_sequence"
    if ref_col not in df.columns:
        raise ValueError("reference_sequence column missing.")
    alt_col = find_alt_column(list(df.columns))
    if alt_col is None:
        raise ValueError("No variant/mutated sequence column found.")

    ref_sequences = []
    alt_sequences = []
    variant_infos = []
    for ref, alt in zip(df[ref_col].astype(str), df[alt_col].astype(str)):
        ref = normalize_sequence(ref)
        alt = normalize_sequence(alt)
        ref_sequences.append(ref)
        alt_sequences.append(alt)
        variant_infos.append(compute_variant_info(ref, alt))
    return ref_sequences, alt_sequences, variant_infos


def build_window_sequences(
    ref_sequences: List[str],
    alt_sequences: List[str],
    variant_infos: List[Dict[str, float]],
    window_bp: int,
) -> Tuple[List[str], List[str], List[int]]:
    ref_windows = []
    alt_windows = []
    in_window_flags = []
    for ref, alt, info in zip(ref_sequences, alt_sequences, variant_infos):
        center = int(info["center"])
        ref_windows.append(slice_window(ref, center, window_bp))
        alt_windows.append(slice_window(alt, center, window_bp))
        start, end = compute_window_bounds(center, len(ref), window_bp)
        in_window = int(info["start"] >= start and info["end"] <= end)
        in_window_flags.append(in_window)
    return ref_windows, alt_windows, in_window_flags


def build_variant_feature_matrix(
    variant_infos: List[Dict[str, float]],
    in_window_by_window: List[List[int]],
) -> np.ndarray:
    features = []
    for idx, info in enumerate(variant_infos):
        ref_len = info["ref_len"] if info["ref_len"] > 0 else 1.0
        center_ratio = info["center"] / max(ref_len - 1.0, 1.0)
        row = [
            info["len_delta"],
            info["mismatch_count"],
            info["span_len"],
            center_ratio,
        ]
        variant_type = info["variant_type"]
        row.extend(
            [
                1.0 if variant_type == "snv" else 0.0,
                1.0 if variant_type == "ins" else 0.0,
                1.0 if variant_type == "del" else 0.0,
                1.0 if variant_type == "complex" else 0.0,
            ]
        )
        for window_flags in in_window_by_window:
            row.append(float(window_flags[idx]))
        features.append(row)
    return np.array(features, dtype=np.float32)


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
    start, end = compute_window_bounds(center, len(seq), window_bp)
    return seq[start : end + 1]


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


def concat_features(
    embeddings_list: List[np.ndarray],
    extra_features: Optional[np.ndarray],
) -> np.ndarray:
    features = list(embeddings_list)
    if extra_features is not None:
        features.append(extra_features)
    if len(features) == 1:
        return features[0]
    return np.concatenate(features, axis=1)


def compute_pll_delta(
    ref_seq: str,
    alt_seq: str,
    tokenizer,
    lm_model,
    device: torch.device,
    max_length: int,
    max_positions: int,
) -> float:
    if lm_model is None or tokenizer.mask_token_id is None:
        return 0.0
    ref_ids = tokenizer(
        ref_seq, add_special_tokens=False, truncation=True, max_length=max_length
    ).input_ids
    alt_ids = tokenizer(
        alt_seq, add_special_tokens=False, truncation=True, max_length=max_length
    ).input_ids
    if not ref_ids or not alt_ids:
        return 0.0

    min_len = min(len(ref_ids), len(alt_ids))
    diffs = [i for i in range(min_len) if ref_ids[i] != alt_ids[i]]
    if not diffs:
        return 0.0
    if max_positions > 0 and len(diffs) > max_positions:
        diffs = diffs[:max_positions]

    ref_masked = list(ref_ids)
    alt_masked = list(alt_ids)
    for pos in diffs:
        ref_masked[pos] = tokenizer.mask_token_id
        alt_masked[pos] = tokenizer.mask_token_id

    ref_input = torch.tensor([ref_masked], device=device)
    alt_input = torch.tensor([alt_masked], device=device)
    with torch.no_grad():
        ref_logits = lm_model(input_ids=ref_input).logits
        alt_logits = lm_model(input_ids=alt_input).logits

    ref_log_probs = torch.log_softmax(ref_logits[0, diffs, :], dim=-1)
    alt_log_probs = torch.log_softmax(alt_logits[0, diffs, :], dim=-1)
    ref_token_ids = torch.tensor([ref_ids[pos] for pos in diffs], device=device)
    alt_token_ids = torch.tensor([alt_ids[pos] for pos in diffs], device=device)
    ref_score = ref_log_probs.gather(1, ref_token_ids.unsqueeze(1)).sum()
    alt_score = alt_log_probs.gather(1, alt_token_ids.unsqueeze(1)).sum()
    return float(alt_score - ref_score)


def compute_pll_deltas(
    ref_sequences: List[str],
    alt_sequences: List[str],
    tokenizer,
    lm_model,
    device: torch.device,
    max_length: int,
    max_positions: int,
) -> np.ndarray:
    deltas = []
    for ref_seq, alt_seq in tqdm(
        zip(ref_sequences, alt_sequences),
        total=len(ref_sequences),
        desc="PLL delta",
    ):
        delta = compute_pll_delta(
            ref_seq,
            alt_seq,
            tokenizer,
            lm_model,
            device,
            max_length,
            max_positions,
        )
        deltas.append(delta)
    return np.array(deltas, dtype=np.float32).reshape(-1, 1)


def load_split(csv_path: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ["__index_level_0__"] if c in df.columns])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


def build_logreg(classes: np.ndarray, class_weight: Optional[str]) -> LogisticRegression:
    if len(classes) > 2:
        solver = "lbfgs"
        multi_class = "multinomial"
    else:
        solver = "liblinear"
        multi_class = "auto"
    try:
        return LogisticRegression(
            max_iter=2000,
            solver=solver,
            multi_class=multi_class,
            class_weight=class_weight,
            n_jobs=1,
        )
    except TypeError:
        # Older scikit-learn versions do not accept multi_class.
        return LogisticRegression(
            max_iter=2000,
            solver=solver,
            class_weight=class_weight,
            n_jobs=1,
        )


def train_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classifier: str,
    class_weight: Optional[str],
    mlp_hidden: Tuple[int, ...],
    seed: int,
    standardize: bool,
):
    classes = np.unique(y_train)
    if classifier == "mlp":
        base = MLPClassifier(
            hidden_layer_sizes=mlp_hidden,
            max_iter=200,
            random_state=seed,
        )
    else:
        base = build_logreg(classes, class_weight)

    if standardize:
        clf = Pipeline([("scaler", StandardScaler()), ("clf", base)])
    else:
        clf = base

    clf.fit(x_train, y_train)
    return clf


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    y_prob: Optional[np.ndarray] = None,
    report_auroc: bool = False,
    report_auprc: bool = False,
) -> Dict[str, float]:
    if num_classes > 2:
        f1_avg = "macro"
    else:
        f1_avg = "binary"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=f1_avg)),
    }
    if num_classes == 2 and y_prob is not None:
        if report_auroc:
            try:
                metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
            except ValueError:
                metrics["auroc"] = float("nan")
        if report_auprc:
            try:
                metrics["auprc"] = float(average_precision_score(y_true, y_prob))
            except ValueError:
                metrics["auprc"] = float("nan")
    return metrics


def run_dataset(
    dataset_dir: str,
    tokenizer,
    model,
    device: torch.device,
    lm_model,
    batch_size: int,
    max_length: int,
    pooling: str,
    strategy: str,
    window_bps: List[int],
    output_dir: str,
    run_name: str,
    max_samples: Optional[int],
    seed: int,
    preset: Optional[str],
    add_variant_features: bool,
    add_pll_delta: bool,
    pll_max_positions: int,
    classifier: str,
    mlp_hidden: Tuple[int, ...],
    class_weight: Optional[str],
    standardize: bool,
    report_auroc: bool,
    report_auprc: bool,
) -> None:
    split_paths = get_split_paths(dataset_dir)
    if "train" not in split_paths or "test" not in split_paths:
        LOGGER.warning("Skipping %s: missing train/test CSV.", dataset_dir)
        return

    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    approach_tag = build_approach_tag(
        preset,
        model.name_or_path,
        window_bps,
        pooling,
        strategy,
        add_variant_features,
        add_pll_delta,
        classifier,
        mlp_hidden,
        class_weight,
        standardize,
    )
    approach_dir = sanitize_slug(f"{approach_tag}__{run_name}", max_len=120)
    output_path = os.path.join(output_dir, dataset_name, approach_dir)
    os.makedirs(output_path, exist_ok=True)
    log_handler = add_log_handler(os.path.join(output_path, "run.log"))
    try:
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

        train_refs, train_alts, train_infos = collect_variant_infos(train_df)
        test_refs, test_alts, test_infos = collect_variant_infos(test_df)
        val_refs = val_alts = val_infos = None
        if val_df is not None:
            val_refs, val_alts, val_infos = collect_variant_infos(val_df)

        window_bps = sorted(set(window_bps))
        window_coverage = {}
        train_embeddings_list = []
        test_embeddings_list = []
        val_embeddings_list = []
        in_window_train_by_window = []
        in_window_test_by_window = []
        in_window_val_by_window = []

        for window_bp in window_bps:
            LOGGER.info("Encoding window %s for %s", window_bp, dataset_dir)
            ref_train_win, alt_train_win, train_in_window = build_window_sequences(
                train_refs, train_alts, train_infos, window_bp
            )
            ref_test_win, alt_test_win, test_in_window = build_window_sequences(
                test_refs, test_alts, test_infos, window_bp
            )
            train_embeddings_list.append(
                encode_pairs(
                    ref_train_win,
                    alt_train_win,
                    tokenizer,
                    model,
                    device,
                    batch_size,
                    max_length,
                    pooling,
                    strategy,
                )
            )
            test_embeddings_list.append(
                encode_pairs(
                    ref_test_win,
                    alt_test_win,
                    tokenizer,
                    model,
                    device,
                    batch_size,
                    max_length,
                    pooling,
                    strategy,
                )
            )
            in_window_train_by_window.append(train_in_window)
            in_window_test_by_window.append(test_in_window)
            window_coverage[str(window_bp)] = {
                "train": float(np.mean(train_in_window)),
                "test": float(np.mean(test_in_window)),
            }

            if val_df is not None and val_refs is not None and val_alts is not None:
                ref_val_win, alt_val_win, val_in_window = build_window_sequences(
                    val_refs, val_alts, val_infos, window_bp
                )
                val_embeddings_list.append(
                    encode_pairs(
                        ref_val_win,
                        alt_val_win,
                        tokenizer,
                        model,
                        device,
                        batch_size,
                        max_length,
                        pooling,
                        strategy,
                    )
                )
                in_window_val_by_window.append(val_in_window)
                window_coverage[str(window_bp)]["validation"] = float(
                    np.mean(val_in_window)
                )

        extra_train = extra_test = extra_val = None
        if add_variant_features:
            extra_train = build_variant_feature_matrix(
                train_infos, in_window_train_by_window
            )
            extra_test = build_variant_feature_matrix(
                test_infos, in_window_test_by_window
            )
            if val_df is not None and val_infos is not None:
                extra_val = build_variant_feature_matrix(
                    val_infos, in_window_val_by_window
                )

        if add_pll_delta:
            if lm_model is None:
                LOGGER.warning("PLL delta requested but LM model is unavailable.")
            else:
                pll_train = compute_pll_deltas(
                    train_refs,
                    train_alts,
                    tokenizer,
                    lm_model,
                    device,
                    max_length,
                    pll_max_positions,
                )
                pll_test = compute_pll_deltas(
                    test_refs,
                    test_alts,
                    tokenizer,
                    lm_model,
                    device,
                    max_length,
                    pll_max_positions,
                )
                extra_train = (
                    pll_train
                    if extra_train is None
                    else np.concatenate([extra_train, pll_train], axis=1)
                )
                extra_test = (
                    pll_test
                    if extra_test is None
                    else np.concatenate([extra_test, pll_test], axis=1)
                )
                if val_df is not None and val_refs is not None and val_alts is not None:
                    pll_val = compute_pll_deltas(
                        val_refs,
                        val_alts,
                        tokenizer,
                        lm_model,
                        device,
                        max_length,
                        pll_max_positions,
                    )
                    extra_val = (
                        pll_val
                        if extra_val is None
                        else np.concatenate([extra_val, pll_val], axis=1)
                    )

        x_train = concat_features(train_embeddings_list, extra_train)
        x_test = concat_features(test_embeddings_list, extra_test)
        x_val = None
        if val_embeddings_list:
            x_val = concat_features(val_embeddings_list, extra_val)

        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_labels)
        y_test = label_encoder.transform(test_labels)
        y_val = label_encoder.transform(val_labels) if val_labels is not None else None

        pos_index = None
        if "pathogenic" in label_encoder.classes_:
            pos_index = int(label_encoder.transform(["pathogenic"])[0])

        LOGGER.info("Training classifier for %s", dataset_dir)
        clf = train_classifier(
            x_train,
            y_train,
            classifier=classifier,
            class_weight=class_weight,
            mlp_hidden=mlp_hidden,
            seed=seed,
            standardize=standardize,
        )
        test_pred = clf.predict(x_test)
        test_prob = None
        if (report_auroc or report_auprc) and hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(x_test)
            if probs.ndim == 2 and probs.shape[1] > 1:
                index = pos_index if pos_index is not None else 1
                test_prob = probs[:, index]
        metrics = {
            "test": evaluate(
                y_test,
                test_pred,
                len(label_encoder.classes_),
                y_prob=test_prob,
                report_auroc=report_auroc,
                report_auprc=report_auprc,
            )
        }
        if x_val is not None and y_val is not None:
            val_pred = clf.predict(x_val)
            val_prob = None
            if (report_auroc or report_auprc) and hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(x_val)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    index = pos_index if pos_index is not None else 1
                    val_prob = probs[:, index]
            metrics["validation"] = evaluate(
                y_val,
                val_pred,
                len(label_encoder.classes_),
                y_prob=val_prob,
                report_auroc=report_auroc,
                report_auprc=report_auprc,
            )

        model_path = os.path.join(output_path, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "classifier": clf,
                    "label_encoder": label_encoder,
                    "settings": {
                        "preset": preset,
                        "model": model.name_or_path,
                        "window_bps": window_bps,
                        "pooling": pooling,
                        "strategy": strategy,
                        "max_length": max_length,
                        "batch_size": batch_size,
                        "add_variant_features": add_variant_features,
                        "add_pll_delta": add_pll_delta,
                        "pll_max_positions": pll_max_positions,
                        "classifier": classifier,
                        "mlp_hidden": mlp_hidden if classifier == "mlp" else None,
                        "class_weight": class_weight,
                        "standardize": standardize,
                        "run_name": run_name,
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
                    "window_coverage": window_coverage,
                    "settings": {
                        "preset": preset,
                        "model": model.name_or_path,
                        "window_bps": window_bps,
                        "pooling": pooling,
                        "strategy": strategy,
                        "max_length": max_length,
                        "batch_size": batch_size,
                        "add_variant_features": add_variant_features,
                        "add_pll_delta": add_pll_delta,
                        "pll_max_positions": pll_max_positions,
                        "classifier": classifier,
                        "mlp_hidden": mlp_hidden if classifier == "mlp" else None,
                        "class_weight": class_weight,
                        "standardize": standardize,
                        "report_auroc": report_auroc,
                        "report_auprc": report_auprc,
                        "run_name": run_name,
                    },
                },
                f,
                indent=2,
            )
        LOGGER.info("Saved metrics to %s", metrics_path)
    finally:
        LOGGER.removeHandler(log_handler)
        log_handler.close()


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
    parser.add_argument(
        "--window_bps",
        default=None,
        help="Comma-separated window sizes (overrides --window_bp).",
    )
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--strategy",
        choices=["concat", "diff", "concat_diff", "ref_only", "alt_only"],
        default="concat",
    )
    parser.add_argument(
        "--classifier",
        choices=["logreg", "mlp"],
        default="logreg",
    )
    parser.add_argument(
        "--mlp_hidden",
        default="256,128",
        help="Comma-separated hidden layer sizes for MLP.",
    )
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--add_variant_features", action="store_true")
    parser.add_argument("--add_pll_delta", action="store_true")
    parser.add_argument(
        "--pll_max_positions",
        type=int,
        default=16,
        help="Max differing token positions to score per sample.",
    )
    parser.add_argument("--no_class_weight", action="store_true")
    parser.add_argument("--report_auroc", action="store_true")
    parser.add_argument("--report_auprc", action="store_true")
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
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    LOGGER.info("Run name: %s", run_name)
    LOGGER.info("Output base dir: %s", args.output_dir)
    LOGGER.info("Args: %s", vars(args))
    if args.preset:
        LOGGER.info("Preset %s -> %s", args.preset, args.model)

    window_bps = parse_window_bps(args.window_bps, args.window_bp)
    class_weight = None if args.no_class_weight else "balanced"
    mlp_hidden = parse_hidden_sizes(args.mlp_hidden) if args.classifier == "mlp" else ()

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

    lm_model = None
    if args.add_pll_delta:
        LOGGER.info("Loading LM head for PLL delta.")
        try:
            lm_model = AutoModelForMaskedLM.from_pretrained(
                args.model, trust_remote_code=args.trust_remote_code
            ).to(device)
        except Exception as exc:
            LOGGER.warning("PLL delta disabled: %s", exc)
            lm_model = None

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
            lm_model=lm_model,
            batch_size=args.batch_size,
            max_length=max_length,
            pooling=args.pooling,
            strategy=args.strategy,
            window_bps=window_bps,
            output_dir=args.output_dir,
            run_name=run_name,
            max_samples=args.max_samples,
            seed=args.seed,
            preset=args.preset,
            add_variant_features=args.add_variant_features,
            add_pll_delta=args.add_pll_delta,
            pll_max_positions=args.pll_max_positions,
            classifier=args.classifier,
            mlp_hidden=mlp_hidden,
            class_weight=class_weight,
            standardize=args.standardize,
            report_auroc=args.report_auroc,
            report_auprc=args.report_auprc,
        )


if __name__ == "__main__":
    main()
