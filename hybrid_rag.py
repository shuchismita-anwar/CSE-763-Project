
import argparse
import datetime
import inspect
import json
import logging
import os
import random
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BertConfig,
)

try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None


DATASET_DIRS = [
    "dataset/wanglab___kegg",
    "dataset/wanglab___variant_effect_coding",
    "dataset/wanglab___variant_effect_non_snv",
]

DNA_PRESETS = {
    "dnabert2": {"model": "zhihan1996/DNABERT-2-117M", "trust_remote_code": True},
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
}

LLM_PRESETS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}

ALT_SEQUENCE_CANDIDATES = [
    "variant_sequence",
    "mutated_sequence",
    "alt_sequence",
]

VALID_BASES_PATTERN = re.compile(r"[^ACGTN]")
ANSWER_RE = re.compile(r"final answer\s*:\s*(.+)", re.IGNORECASE)
NON_PATHOGENIC_RE = re.compile(r"\b(non[- ]?pathogenic|not pathogenic)\b")
VARIANT_GROUPS = ["snv", "mnp", "indel"]

LOGGER = logging.getLogger("hybrid_rag")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sanitize_slug(text: str, max_len: int = 120) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    if not text:
        text = "run"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text


def short_model_tag(model_name: str) -> str:
    tag = model_name.split("/")[-1]
    return sanitize_slug(tag, max_len=32)


def build_approach_tag(
    dna_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    window_bps: List[int],
    pooling: str,
    strategy: str,
    retrieval_strategy: str,
    add_variant_features: bool,
    add_pll_delta: bool,
    match_variant_type: bool,
    match_length_bucket: bool,
    enforce_majority_vote: bool,
    include_variant_info: bool,
    include_label_list: bool,
    include_answer_text: bool,
    max_example_chars: int,
    use_chat_template: bool,
) -> str:
    dna_tag = dna_settings.get("preset") or short_model_tag(
        dna_settings.get("model", "dna")
    )
    llm_tag = llm_settings.get("preset") or short_model_tag(
        llm_settings.get("model", "llm")
    )
    window_tag = "w" + "-".join(str(val) for val in window_bps)
    parts = [
        dna_tag,
        llm_tag,
        window_tag,
        f"pool-{pooling}",
        f"strat-{strategy}",
        f"retr-{retrieval_strategy}",
        f"ex{max_example_chars}",
    ]
    if add_variant_features:
        parts.append("varfeat")
    if add_pll_delta:
        parts.append("pll")
    if match_variant_type:
        parts.append("matchtype")
    if match_length_bucket:
        parts.append("matchlen")
    if enforce_majority_vote:
        parts.append("majvote")
    if include_variant_info:
        parts.append("varinfo")
    if include_label_list:
        parts.append("labels")
    if include_answer_text:
        parts.append("answers")
    if use_chat_template:
        parts.append("chat")
    return sanitize_slug("__".join(parts), max_len=120)


def add_log_handler(log_path: str):
    handler = logging.FileHandler(log_path)
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    return handler


def normalize_label(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text.strip())


def normalize_sequence(seq: str) -> str:
    seq = seq.upper()
    return VALID_BASES_PATTERN.sub("N", seq)


def parse_window_bps(window_bps: Optional[str], window_bp: int) -> List[int]:
    if window_bps:
        values = [int(val.strip()) for val in window_bps.split(",") if val.strip()]
        if values:
            return values
    return [window_bp]


def find_alt_column(columns: List[str]) -> Optional[str]:
    lower_map = {col.lower(): col for col in columns}
    for candidate in ALT_SEQUENCE_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


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


def extract_answer_text(row: pd.Series) -> str:
    if "answer" not in row or pd.isna(row["answer"]):
        return ""
    text = str(row["answer"])
    if ";" in text:
        return text.split(";", 1)[1].strip()
    return ""


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


def get_split_paths(dataset_dir: str) -> Dict[str, str]:
    splits = {}
    for split_name in ["train", "validation", "val", "test"]:
        path = os.path.join(dataset_dir, f"{split_name}.csv")
        if os.path.exists(path):
            key = "validation" if split_name == "val" else split_name
            splits[key] = path
    return splits


def load_split(csv_path: str, max_samples: Optional[int], seed: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=[c for c in ["__index_level_0__"] if c in df.columns])
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return df


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


def length_bucket(len_delta: int) -> str:
    abs_len = abs(len_delta)
    if abs_len == 0:
        return "0"
    if abs_len == 1:
        return "1"
    if abs_len <= 5:
        return "2-5"
    if abs_len <= 20:
        return "6-20"
    return "21+"


def compute_variant_info(ref: str, alt: str) -> Dict[str, object]:
    start, end, mismatch_count, len_delta = find_variant_span(ref, alt)
    center = find_variant_center(ref, alt)
    span_len = max(1, end - start + 1)
    if len_delta > 0:
        variant_type = "ins"
        variant_group = "indel"
    elif len_delta < 0:
        variant_type = "del"
        variant_group = "indel"
    elif mismatch_count == 1:
        variant_type = "snv"
        variant_group = "snv"
    else:
        variant_type = "mnp"
        variant_group = "mnp"
    return {
        "center": center,
        "span_len": span_len,
        "mismatch_count": mismatch_count,
        "len_delta": len_delta,
        "variant_type": variant_type,
        "variant_group": variant_group,
        "length_bucket": length_bucket(len_delta),
    }


def collect_variant_infos(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    ref_col = "reference_sequence"
    if ref_col not in df.columns:
        raise ValueError("reference_sequence column missing.")
    alt_col = find_alt_column(list(df.columns))
    if alt_col is None:
        raise ValueError("No variant/mutated sequence column found.")

    refs = []
    alts = []
    infos = []
    for ref, alt in zip(df[ref_col].astype(str), df[alt_col].astype(str)):
        ref = normalize_sequence(ref)
        alt = normalize_sequence(alt)
        refs.append(ref)
        alts.append(alt)
        infos.append(compute_variant_info(ref, alt))
    return refs, alts, infos


def build_window_sequences(
    refs: List[str],
    alts: List[str],
    infos: List[Dict[str, object]],
    window_bp: int,
) -> Tuple[List[str], List[str], List[float]]:
    ref_windows = []
    alt_windows = []
    in_window = []
    for ref, alt, info in zip(refs, alts, infos):
        center = int(info["center"])
        start, end = compute_window_bounds(center, len(ref), window_bp)
        ref_windows.append(ref[start : end + 1])
        alt_windows.append(alt[start : end + 1])
        span_len = int(info["span_len"])
        in_window.append(1.0 if span_len <= (end - start + 1) else 0.0)
    return ref_windows, alt_windows, in_window


def build_variant_feature_matrix(
    infos: List[Dict[str, object]],
    window_bps: List[int],
    in_window_by_window: List[List[float]],
) -> np.ndarray:
    features = []
    scale = 100.0
    for idx, info in enumerate(infos):
        len_delta = float(info["len_delta"])
        mismatch_count = float(info["mismatch_count"])
        span_len = float(info["span_len"])
        abs_len = abs(len_delta)
        feats = [
            max(min(len_delta / scale, 1.0), -1.0),
            min(abs_len / scale, 1.0),
            min(mismatch_count / scale, 1.0),
            min(span_len / scale, 1.0),
        ]
        group = str(info["variant_group"])
        feats.extend([1.0 if group == val else 0.0 for val in VARIANT_GROUPS])
        for w_idx, window_bp in enumerate(window_bps):
            span_ratio = min(span_len / max(window_bp, 1), 1.0)
            feats.append(span_ratio)
            feats.append(float(in_window_by_window[w_idx][idx]))
        features.append(feats)
    return np.array(features, dtype=np.float32)


def format_variant_info(info: Dict[str, object]) -> str:
    len_delta = int(info["len_delta"])
    mismatch_count = int(info["mismatch_count"])
    span_len = int(info["span_len"])
    variant_type = str(info["variant_type"]).upper()
    sign = "+" if len_delta > 0 else ""
    return (
        f"type={variant_type}, len_delta={sign}{len_delta}, "
        f"mismatches={mismatch_count}, span={span_len}"
    )


def majority_vote_label(labels: List[str]) -> Tuple[Optional[str], Dict[str, int]]:
    counts = {"benign": 0, "pathogenic": 0}
    for label in labels:
        norm = normalize_label(label)
        if norm in counts:
            counts[norm] += 1
    if counts["benign"] == 0 and counts["pathogenic"] == 0:
        return None, counts
    if counts["benign"] == counts["pathogenic"]:
        return None, counts
    return ("benign" if counts["benign"] > counts["pathogenic"] else "pathogenic"), counts


def filter_neighbors(
    candidate_indices: List[int],
    train_infos: List[Dict[str, object]],
    test_info: Dict[str, object],
    match_variant_type: bool,
    match_length_bucket: bool,
    k: int,
) -> List[int]:
    if not candidate_indices or k <= 0:
        return []
    filtered = []
    target_group = str(test_info["variant_group"])
    target_bucket = str(test_info["length_bucket"])
    for idx in candidate_indices:
        info = train_infos[idx]
        if match_variant_type and str(info["variant_group"]) != target_group:
            continue
        if match_length_bucket and str(info["length_bucket"]) != target_bucket:
            continue
        filtered.append(idx)
        if len(filtered) >= k:
            return filtered
    for idx in candidate_indices:
        if idx in filtered:
            continue
        filtered.append(idx)
        if len(filtered) >= k:
            break
    return filtered


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


def load_dna_model(model_name: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token

    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    except ValueError as exc:
        msg = str(exc)
        if "config_class" in msg and trust_remote_code:
            LOGGER.warning(
                "Config class mismatch detected; retrying with BertConfig (no remote code)."
            )
            config = BertConfig.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name, trust_remote_code=False, config=config
            )
        elif "Unrecognized configuration class" in msg and "AutoModel" in msg:
            LOGGER.warning(
                "AutoModel unsupported; trying AutoModelForMaskedLM then AutoModelForCausalLM."
            )
            try:
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
            except Exception as inner_exc:
                LOGGER.warning("AutoModelForMaskedLM failed: %s", inner_exc)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
        else:
            raise

    return tokenizer, model


def build_kegg_prefix(
    label_list: List[str],
    include_label_list: bool,
    retrieved: List[Dict[str, str]],
) -> str:
    parts = [
        "You are a biomedical assistant. Choose exactly one disease label.",
        "Output format: Final answer: <DISEASE_NAME>",
        "Output only the final answer line. Do not add explanations.",
    ]
    if include_label_list:
        parts.append(f"Labels: {', '.join(label_list)}")
    if retrieved:
        parts.append("Retrieved examples:")
        for idx, ex in enumerate(retrieved, start=1):
            parts.append(f"{idx}. Input: {ex['text']}")
            parts.append(f"   Label: {ex['label']}")
    parts.append("Input:")
    return "\n".join(parts)


def build_vep_prefix(
    retrieved: List[Dict[str, str]],
    vote_summary: Optional[str],
    enforce_majority_vote: bool,
) -> str:
    parts = [
        "You are a biomedical assistant. Classify the variant as BENIGN or PATHOGENIC.",
        "Output format: Final answer: BENIGN or Final answer: PATHOGENIC",
        "Output only the final answer line. Do not add explanations.",
    ]
    if enforce_majority_vote:
        parts.append(
            "Decision rule: use the majority label from the retrieved examples."
        )
    if vote_summary:
        parts.append(f"Label counts: {vote_summary}")
    if retrieved:
        parts.append("Retrieved examples:")
        for idx, ex in enumerate(retrieved, start=1):
            parts.append(f"{idx}. Input: {ex['text']}")
            if ex.get("variant_info"):
                parts.append(f"   Variant: {ex['variant_info']}")
            label = ex["label"].upper()
            parts.append(f"   Label: {label}")
            if ex.get("rationale"):
                parts.append(f"   Info: {ex['rationale']}")
    parts.append("Input:")
    return "\n".join(parts)


def extract_prediction(text: str, label_map: Dict[str, str]) -> str:
    match = ANSWER_RE.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.splitlines()[0]
    normalized = normalize_label(candidate)

    if normalized in ("benign", "pathogenic"):
        return normalized
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
    return sorted(set(train_labels))


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


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."


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


def evaluate(y_true: List[str], y_pred: List[str], num_classes: int) -> Dict[str, float]:
    unique_labels = sorted(set(y_true) | set(y_pred))
    unknown_rate = 0.0
    if y_pred:
        unknown_rate = float(sum(label == "unknown" for label in y_pred) / len(y_pred))

    if len(unique_labels) <= 2 and set(unique_labels) <= {"benign", "pathogenic"}:
        f1_value = f1_score(y_true, y_pred, average="binary", pos_label="pathogenic")
    else:
        f1_value = f1_score(y_true, y_pred, average="macro")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_value),
        "unknown_rate": unknown_rate,
    }


def apply_dna_preset(args: argparse.Namespace) -> None:
    if not args.dna_preset:
        return
    preset = DNA_PRESETS.get(args.dna_preset)
    if preset is None:
        raise ValueError(f"Unknown DNA preset: {args.dna_preset}")
    args.dna_model = preset["model"]
    if preset.get("trust_remote_code"):
        args.dna_trust_remote_code = True


def apply_llm_preset(args: argparse.Namespace) -> None:
    if not args.llm_preset:
        return
    model_name = LLM_PRESETS.get(args.llm_preset)
    if not model_name:
        raise ValueError(f"Unknown LLM preset: {args.llm_preset}")
    args.llm_model = model_name


def require_bitsandbytes() -> None:
    if BitsAndBytesConfig is None:
        raise RuntimeError(
            "bitsandbytes is required for 4-bit/8-bit loading. "
            "Install with: pip install bitsandbytes"
        )


def parse_k_values(k_value: str, k_values: Optional[str]) -> List[int]:
    if k_values:
        return [int(val.strip()) for val in k_values.split(",") if val.strip()]
    return [int(k_value)]


def format_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return prompt


def build_retrieved_examples(
    train_df: pd.DataFrame,
    train_labels: List[str],
    indices: List[int],
    text_col: str,
    include_answer_text: bool,
    max_example_chars: int,
    dataset_dir: str,
    train_infos: Optional[List[Dict[str, object]]],
    include_variant_info: bool,
) -> List[Dict[str, str]]:
    examples = []
    for idx in indices:
        row = train_df.iloc[idx]
        text = truncate_text(str(row[text_col]), max_example_chars)
        label = train_labels[idx]
        entry = {"text": text, "label": label}
        if include_variant_info and train_infos is not None:
            entry["variant_info"] = format_variant_info(train_infos[idx])
        if include_answer_text and "variant_effect" in dataset_dir:
            rationale = truncate_text(extract_answer_text(row), max_example_chars)
            if rationale:
                entry["rationale"] = rationale
        examples.append(entry)
    return examples


def run_dataset(
    dataset_dir: str,
    dna_tokenizer,
    dna_model,
    dna_lm_model,
    dna_device: torch.device,
    dna_batch_size: int,
    dna_max_length: int,
    pooling: str,
    strategy: str,
    window_bps: List[int],
    retrieval_strategy: Optional[str],
    add_variant_features: bool,
    add_pll_delta: bool,
    pll_max_positions: int,
    match_variant_type: bool,
    match_length_bucket: bool,
    retrieval_candidate_mult: int,
    enforce_majority_vote: bool,
    include_variant_info: bool,
    llm_tokenizer,
    llm_model,
    llm_device: torch.device,
    max_new_tokens: int,
    max_input_tokens: int,
    use_chat_template: bool,
    include_label_list: bool,
    include_answer_text: bool,
    max_example_chars: int,
    split: str,
    train_max_samples: Optional[int],
    test_max_samples: Optional[int],
    seed: int,
    k_values: List[int],
    output_dir: str,
    run_name: str,
    dna_settings: Dict[str, str],
    llm_settings: Dict[str, str],
    text_column: Optional[str],
) -> None:
    split_paths = get_split_paths(dataset_dir)
    if "train" not in split_paths or split not in split_paths:
        LOGGER.warning("Skipping %s: missing train/%s CSV.", dataset_dir, split)
        return

    train_df = load_split(split_paths["train"], train_max_samples, seed)
    test_df = load_split(split_paths[split], test_max_samples, seed)
    train_df, train_labels = parse_labels(train_df, dataset_dir)
    test_df, test_labels = parse_labels(test_df, dataset_dir)

    text_col = get_text_column(test_df, text_column)
    train_text_col = get_text_column(train_df, text_column)

    LOGGER.info("Preparing DNA embeddings for %s", dataset_dir)
    ref_train, alt_train, train_infos = collect_variant_infos(train_df)
    ref_test, alt_test, test_infos = collect_variant_infos(test_df)

    is_variant_effect = "variant_effect" in dataset_dir
    window_bps = sorted(set(window_bps))
    if is_variant_effect and len(window_bps) == 1:
        window_bps = sorted(set(window_bps + [1024, 2048]))
        LOGGER.info("Using multi-scale windows for VEP: %s", window_bps)

    effective_strategy = retrieval_strategy
    if effective_strategy is None:
        effective_strategy = "diff" if is_variant_effect else strategy

    use_variant_features = add_variant_features and is_variant_effect
    use_variant_info_prompt = include_variant_info and is_variant_effect
    apply_match_variant_type = match_variant_type and is_variant_effect
    apply_match_length_bucket = match_length_bucket and is_variant_effect

    train_embeddings_list = []
    test_embeddings_list = []
    in_window_train_by_window = []
    in_window_test_by_window = []
    for window_bp in window_bps:
        ref_train_win, alt_train_win, train_in_window = build_window_sequences(
            ref_train, alt_train, train_infos, window_bp
        )
        ref_test_win, alt_test_win, test_in_window = build_window_sequences(
            ref_test, alt_test, test_infos, window_bp
        )
        train_embeddings_list.append(
            encode_pairs(
                ref_train_win,
                alt_train_win,
                dna_tokenizer,
                dna_model,
                dna_device,
                dna_batch_size,
                dna_max_length,
                pooling,
                effective_strategy,
            )
        )
        test_embeddings_list.append(
            encode_pairs(
                ref_test_win,
                alt_test_win,
                dna_tokenizer,
                dna_model,
                dna_device,
                dna_batch_size,
                dna_max_length,
                pooling,
                effective_strategy,
            )
        )
        in_window_train_by_window.append(train_in_window)
        in_window_test_by_window.append(test_in_window)

    extra_train = extra_test = None
    if use_variant_features:
        extra_train = build_variant_feature_matrix(
            train_infos, window_bps, in_window_train_by_window
        )
        extra_test = build_variant_feature_matrix(
            test_infos, window_bps, in_window_test_by_window
        )

    if add_pll_delta:
        if dna_lm_model is None:
            LOGGER.warning("PLL delta requested but LM model is unavailable.")
        else:
            pll_train = compute_pll_deltas(
                ref_train,
                alt_train,
                dna_tokenizer,
                dna_lm_model,
                dna_device,
                dna_max_length,
                pll_max_positions,
            )
            pll_test = compute_pll_deltas(
                ref_test,
                alt_test,
                dna_tokenizer,
                dna_lm_model,
                dna_device,
                dna_max_length,
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

    x_train = concat_features(train_embeddings_list, extra_train).astype(np.float32)
    x_test = concat_features(test_embeddings_list, extra_test).astype(np.float32)

    k_values = sorted(set(k_values))
    max_k = max(k_values) if k_values else 0
    neighbor_indices = None
    if max_k > 0:
        candidate_k = max_k * max(1, retrieval_candidate_mult)
        LOGGER.info(
            "Computing nearest neighbors (k=%s, candidate_k=%s) for %s",
            max_k,
            candidate_k,
            dataset_dir,
        )
        nn = NearestNeighbors(n_neighbors=candidate_k, metric="cosine")
        nn.fit(x_train)
        _, neighbor_indices = nn.kneighbors(x_test, n_neighbors=candidate_k)

    label_list = get_label_list(train_labels)
    label_map = {normalize_label(label): label for label in label_list}
    normalized_label_set = {normalize_label(label) for label in label_list}
    is_binary_task = normalized_label_set <= {"benign", "pathogenic"} and len(
        normalized_label_set
    ) <= 2
    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    approach_tag = build_approach_tag(
        dna_settings,
        llm_settings,
        window_bps,
        pooling,
        strategy,
        effective_strategy,
        use_variant_features,
        add_pll_delta,
        apply_match_variant_type,
        apply_match_length_bucket,
        enforce_majority_vote and is_binary_task,
        use_variant_info_prompt,
        include_label_list,
        include_answer_text,
        max_example_chars,
        use_chat_template,
    )
    approach_dir = sanitize_slug(f"{approach_tag}__{run_name}", max_len=140)
    dataset_output_dir = os.path.join(output_dir, dataset_name, approach_dir)
    os.makedirs(dataset_output_dir, exist_ok=True)
    log_handler = add_log_handler(os.path.join(dataset_output_dir, "run.log"))

    try:
        for k in k_values:
            LOGGER.info("Running hybrid RAG for %s (k=%s)", dataset_dir, k)
            predictions = []
            raw_outputs = []
            neighbor_labels = []
            neighbor_ids = []
            neighbor_variant_types = []
            neighbor_len_deltas = []
            vote_labels = []
            vote_summaries = []

            for idx, text in enumerate(
                tqdm(test_df[text_col].astype(str).tolist(), desc=f"Generating k={k}")
            ):
                indices = []
                if k > 0 and neighbor_indices is not None:
                    candidate_indices = neighbor_indices[idx].tolist()
                    indices = filter_neighbors(
                        candidate_indices,
                        train_infos,
                        test_infos[idx],
                        apply_match_variant_type,
                        apply_match_length_bucket,
                        k,
                    )
                retrieved = build_retrieved_examples(
                    train_df,
                    train_labels,
                    indices,
                    train_text_col,
                    include_answer_text,
                    max_example_chars,
                    dataset_dir,
                    train_infos,
                    use_variant_info_prompt,
                )
                vote_label, vote_counts = majority_vote_label(
                    [train_labels[i] for i in indices]
                )
                vote_summary = None
                if is_binary_task:
                    vote_summary = (
                        f"BENIGN={vote_counts['benign']} "
                        f"PATHOGENIC={vote_counts['pathogenic']}"
                    )

                if "kegg" in dataset_dir:
                    prefix = build_kegg_prefix(label_list, include_label_list, retrieved)
                else:
                    prefix = build_vep_prefix(
                        retrieved,
                        vote_summary,
                        enforce_majority_vote and is_binary_task,
                    )

                input_text = text
                if use_variant_info_prompt:
                    input_text = (
                        f"Variant info: {format_variant_info(test_infos[idx])}\n{text}"
                    )

                prompt = truncate_question(
                    prefix,
                    input_text,
                    llm_tokenizer,
                    max_prompt_tokens=max_input_tokens - max_new_tokens,
                )
                formatted = format_prompt(llm_tokenizer, prompt, use_chat_template)
                inputs = llm_tokenizer(
                    formatted,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                )
                inputs = {k: v.to(llm_device) for k, v in inputs.items()}
                with torch.no_grad():
                    output_ids = llm_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=0.0,
                        top_p=1.0,
                        pad_token_id=llm_tokenizer.pad_token_id,
                    )
                generated = output_ids[0][inputs["input_ids"].shape[-1] :]
                decoded = llm_tokenizer.decode(generated, skip_special_tokens=True)
                raw_outputs.append(decoded)
                pred = extract_prediction(decoded, label_map)
                final_pred = pred
                if is_binary_task and vote_label is not None:
                    if enforce_majority_vote:
                        final_pred = vote_label
                    elif pred == "unknown":
                        final_pred = vote_label
                predictions.append(final_pred)
                neighbor_ids.append(indices)
                neighbor_labels.append([train_labels[i] for i in indices])
                neighbor_variant_types.append(
                    [str(train_infos[i]["variant_type"]) for i in indices]
                )
                neighbor_len_deltas.append(
                    [int(train_infos[i]["len_delta"]) for i in indices]
                )
                vote_labels.append(vote_label or "unknown")
                vote_summaries.append(vote_summary or "")

            true_labels = [normalize_label(label) for label in test_labels]
            pred_labels = [normalize_label(label) for label in predictions]
            metrics = evaluate(true_labels, pred_labels, len(label_list))

            output_path = os.path.join(dataset_output_dir, f"k_{k}")
            os.makedirs(output_path, exist_ok=True)

            metrics_path = os.path.join(output_path, "metrics.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "dataset": dataset_name,
                        "k": k,
                        "metrics": metrics,
                        "settings": {
                            "dna": dna_settings,
                            "llm": llm_settings,
                            "window_bps": window_bps,
                            "pooling": pooling,
                            "strategy": strategy,
                            "retrieval_strategy": effective_strategy,
                            "max_input_tokens": max_input_tokens,
                            "max_new_tokens": max_new_tokens,
                            "include_label_list": include_label_list,
                            "include_answer_text": include_answer_text,
                            "include_variant_info": use_variant_info_prompt,
                            "max_example_chars": max_example_chars,
                            "split": split,
                            "text_column": text_column,
                            "add_variant_features": use_variant_features,
                            "add_pll_delta": add_pll_delta,
                            "pll_max_positions": pll_max_positions,
                            "match_variant_type": apply_match_variant_type,
                            "match_length_bucket": apply_match_length_bucket,
                            "retrieval_candidate_mult": retrieval_candidate_mult,
                            "enforce_majority_vote": enforce_majority_vote and is_binary_task,
                            "run_name": run_name,
                        },
                    },
                    f,
                    indent=2,
                )

            preds_df = pd.DataFrame(
                {
                    "text": test_df[text_col].astype(str),
                    "label": true_labels,
                    "prediction": pred_labels,
                    "raw_output": raw_outputs,
                    "vote_label": vote_labels,
                    "vote_summary": vote_summaries,
                    "neighbor_indices": [json.dumps(x) for x in neighbor_ids],
                    "neighbor_labels": [json.dumps(x) for x in neighbor_labels],
                    "neighbor_variant_types": [
                        json.dumps(x) for x in neighbor_variant_types
                    ],
                    "neighbor_len_deltas": [json.dumps(x) for x in neighbor_len_deltas],
                }
            )
            preds_path = os.path.join(output_path, "predictions.csv")
            preds_df.to_csv(preds_path, index=False)

            LOGGER.info("Saved metrics to %s", metrics_path)
            LOGGER.info("Saved predictions to %s", preds_path)
    finally:
        LOGGER.removeHandler(log_handler)
        log_handler.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid DNA-RAG + LLM baselines")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DATASET_DIRS,
        help="Dataset directories containing train/test CSVs.",
    )
    parser.add_argument(
        "--dna_preset",
        choices=sorted(DNA_PRESETS.keys()),
        help="DNA encoder preset (overrides --dna_model).",
    )
    parser.add_argument(
        "--llm_preset",
        choices=sorted(LLM_PRESETS.keys()),
        help="LLM preset (overrides --llm_model).",
    )
    parser.add_argument("--list_presets", action="store_true")
    parser.add_argument(
        "--dna_model",
        default="zhihan1996/DNABERT-2-117M",
        help="HF model name for DNA encoder.",
    )
    parser.add_argument(
        "--llm_model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model name for LLM.",
    )
    parser.add_argument("--dna_batch_size", type=int, default=4)
    parser.add_argument("--dna_max_length", type=int, default=None)
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
        "--retrieval_strategy",
        choices=["concat", "diff", "concat_diff", "ref_only", "alt_only"],
        default=None,
        help="Embedding strategy used for retrieval (defaults to --strategy or diff for VEP).",
    )
    parser.add_argument("--add_variant_features", action="store_true", default=True)
    parser.add_argument(
        "--no_variant_features",
        action="store_true",
        help="Disable explicit variant features for retrieval/prompt.",
    )
    parser.add_argument("--add_pll_delta", action="store_true")
    parser.add_argument(
        "--pll_max_positions",
        type=int,
        default=16,
        help="Max differing token positions to score per sample.",
    )
    parser.add_argument(
        "--retrieval_candidate_mult",
        type=int,
        default=5,
        help="Retrieve k * multiplier candidates before filtering.",
    )
    parser.add_argument(
        "--no_match_variant_type",
        action="store_true",
        help="Disable matching SNV vs indel before retrieval.",
    )
    parser.add_argument(
        "--no_match_length_bucket",
        action="store_true",
        help="Disable matching by length bucket before retrieval.",
    )
    parser.add_argument(
        "--no_enforce_majority_vote",
        action="store_true",
        help="Disable majority-vote enforcement for binary tasks.",
    )
    parser.add_argument(
        "--no_variant_info",
        action="store_true",
        help="Disable adding variant info to prompts.",
    )
    parser.add_argument("--train_max_samples", type=int, default=None)
    parser.add_argument("--test_max_samples", type=int, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument(
        "--k_values",
        default=None,
        help="Comma-separated list of k values (overrides --k).",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--max_input_tokens", type=int, default=None)
    parser.add_argument("--use_chat_template", action="store_true")
    parser.add_argument("--include_label_list", action="store_true")
    parser.add_argument("--include_answer_text", action="store_true")
    parser.add_argument("--max_example_chars", type=int, default=300)
    parser.add_argument("--output_dir", default="outputs/hybrid_rag")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--dna_device", default=None)
    parser.add_argument("--llm_device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dna_trust_remote_code", action="store_true")
    parser.add_argument("--llm_trust_remote_code", action="store_true")
    parser.add_argument("--llm_load_in_8bit", action="store_true")
    parser.add_argument("--llm_load_in_4bit", action="store_true")
    parser.add_argument(
        "--bnb_compute_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--llm_device_map", default=None)
    parser.add_argument("--text_column", default="question")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_presets:
        print("DNA presets:")
        for name, preset in DNA_PRESETS.items():
            print(f"  {name}: {preset['model']}")
        print("LLM presets:")
        for name, model_name in LLM_PRESETS.items():
            print(f"  {name}: {model_name}")
        sys.exit(0)

    apply_dna_preset(args)
    apply_llm_preset(args)
    if args.llm_load_in_4bit and args.llm_load_in_8bit:
        raise ValueError("Choose only one of --llm_load_in_4bit or --llm_load_in_8bit.")

    set_seed(args.seed)

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

    dna_device = torch.device(
        args.dna_device if args.dna_device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    llm_device = torch.device(
        args.llm_device if args.llm_device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    LOGGER.info("Loading DNA model: %s", args.dna_model)
    dna_tokenizer, dna_model = load_dna_model(args.dna_model, args.dna_trust_remote_code)
    dna_model = dna_model.to(dna_device)
    dna_max_length = get_max_length(dna_tokenizer, dna_model, args.dna_max_length)

    dna_lm_model = None
    if args.add_pll_delta:
        LOGGER.info("Loading DNA LM for PLL delta: %s", args.dna_model)
        try:
            dna_lm_model = AutoModelForMaskedLM.from_pretrained(
                args.dna_model, trust_remote_code=args.dna_trust_remote_code
            ).to(dna_device)
            dna_lm_model.eval()
        except Exception as exc:
            LOGGER.warning("PLL delta disabled: %s", exc)
            dna_lm_model = None

    quantization = None
    quant_config = None
    device_map = args.llm_device_map
    if args.llm_load_in_4bit or args.llm_load_in_8bit:
        require_bitsandbytes()
        quantization = "4bit" if args.llm_load_in_4bit else "8bit"
        compute_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.bnb_compute_dtype]
        quant_config = BitsAndBytesConfig(
            load_in_4bit=args.llm_load_in_4bit,
            load_in_8bit=args.llm_load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        if device_map is None:
            device_map = "auto"

    LOGGER.info("Loading LLM: %s", args.llm_model)
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model, trust_remote_code=args.llm_trust_remote_code
    )
    if llm_tokenizer.pad_token_id is None:
        if llm_tokenizer.eos_token_id is not None:
            llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
        elif llm_tokenizer.unk_token_id is not None:
            llm_tokenizer.pad_token_id = llm_tokenizer.unk_token_id

    torch_dtype = "auto" if llm_device.type == "cuda" else torch.float32
    if quant_config is not None:
        torch_dtype = None
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        trust_remote_code=args.llm_trust_remote_code,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_config,
    )
    if device_map is None:
        llm_model = llm_model.to(llm_device)
    llm_model.eval()

    max_input_tokens = get_max_input_tokens(
        llm_tokenizer, llm_model, args.max_input_tokens
    )
    LOGGER.info("Max input tokens: %s", max_input_tokens)

    k_values = parse_k_values(str(args.k), args.k_values)
    window_bps = parse_window_bps(args.window_bps, args.window_bp)
    add_variant_features = args.add_variant_features and not args.no_variant_features
    match_variant_type = not args.no_match_variant_type
    match_length_bucket = not args.no_match_length_bucket
    enforce_majority_vote = not args.no_enforce_majority_vote
    include_variant_info = not args.no_variant_info
    dna_settings = {
        "preset": args.dna_preset,
        "model": args.dna_model,
        "max_length": dna_max_length,
        "batch_size": args.dna_batch_size,
    }
    llm_settings = {
        "preset": args.llm_preset,
        "model": args.llm_model,
        "quantization": quantization,
        "device_map": device_map,
    }

    for dataset_dir in args.datasets:
        if not os.path.isdir(dataset_dir):
            LOGGER.warning("Missing dataset dir: %s", dataset_dir)
            continue
        run_dataset(
            dataset_dir=dataset_dir,
            dna_tokenizer=dna_tokenizer,
            dna_model=dna_model,
            dna_lm_model=dna_lm_model,
            dna_device=dna_device,
            dna_batch_size=args.dna_batch_size,
            dna_max_length=dna_max_length,
            pooling=args.pooling,
            strategy=args.strategy,
            window_bps=window_bps,
            retrieval_strategy=args.retrieval_strategy,
            add_variant_features=add_variant_features,
            add_pll_delta=args.add_pll_delta,
            pll_max_positions=args.pll_max_positions,
            match_variant_type=match_variant_type,
            match_length_bucket=match_length_bucket,
            retrieval_candidate_mult=args.retrieval_candidate_mult,
            enforce_majority_vote=enforce_majority_vote,
            include_variant_info=include_variant_info,
            llm_tokenizer=llm_tokenizer,
            llm_model=llm_model,
            llm_device=llm_device,
            max_new_tokens=args.max_new_tokens,
            max_input_tokens=max_input_tokens,
            use_chat_template=args.use_chat_template,
            include_label_list=args.include_label_list,
            include_answer_text=args.include_answer_text,
            max_example_chars=args.max_example_chars,
            split=args.split,
            train_max_samples=args.train_max_samples,
            test_max_samples=args.test_max_samples,
            seed=args.seed,
            k_values=k_values,
            output_dir=args.output_dir,
            run_name=run_name,
            dna_settings=dna_settings,
            llm_settings=llm_settings,
            text_column=args.text_column,
        )


if __name__ == "__main__":
    main()
