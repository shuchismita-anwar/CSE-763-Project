import os

import pandas as pd

import matplotlib.pyplot as plt


DATASET_DIRS = [
    "dataset/wanglab___kegg",
    "dataset/wanglab___variant_effect_coding",
    "dataset/wanglab___variant_effect_non_snv",
]

LABEL_CANDIDATES = [
    "label",
    "labels",
    "answer",
    "target",
    "class",
    "cleaned_pathogenicity",
]

MAX_TEXT_PLOTS = 4
TEXT_LENGTH_BINS = 50


def find_label_column(columns):
    lower_map = {col.lower(): col for col in columns}
    for candidate in LABEL_CANDIDATES:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def save_bar_plot(counts, title, path):
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Label")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def save_length_hist(lengths, title, path):
    if plt is None:
        return False
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths.dropna().values, bins=TEXT_LENGTH_BINS)
    ax.set_title(title)
    ax.set_xlabel("Length")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def maybe_plot_split(df, label_col, dataset_dir, split_name):
    if plt is None:
        return

    plot_dir = os.path.join(dataset_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    if label_col:
        counts = df[label_col].value_counts(dropna=False)
        label_plot = os.path.join(plot_dir, f"{split_name}_label_counts.png")
        save_bar_plot(counts, f"{split_name} label counts", label_plot)

    text_columns = [
        col
        for col in df.columns
        if df[col].dtype == object
        and col not in {label_col, "__index_level_0__"}
    ]
    for col in text_columns[:MAX_TEXT_PLOTS]:
        lengths = df[col].astype(str).str.len()
        length_plot = os.path.join(plot_dir, f"{split_name}_{col}_lengths.png")
        save_length_hist(lengths, f"{split_name} {col} lengths", length_plot)


def inspect_split(csv_path, dataset_dir, split_name):
    df = pd.read_csv(csv_path)
    label_col = find_label_column(df.columns)

    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    if label_col:
        counts = df[label_col].value_counts(dropna=False)
        print(f"  Label column: {label_col}")
        print("  Label counts:")
        print(counts)
    else:
        print("  Label column: not found")

    print("  Sample rows:")
    print(df.head(3))
    maybe_plot_split(df, label_col, dataset_dir, split_name)


def inspect_dataset(dataset_dir):
    print(f"\nDataset: {dataset_dir}")
    for split_name in ["train", "validation", "test"]:
        csv_path = os.path.join(dataset_dir, f"{split_name}.csv")
        if not os.path.exists(csv_path):
            continue
        print(f"\n Split: {split_name}")
        inspect_split(csv_path, dataset_dir, split_name)


def main():
    if plt is None:
        print("matplotlib not installed; skipping plots. Install with: pip install matplotlib")
    for dataset_dir in DATASET_DIRS:
        if not os.path.isdir(dataset_dir):
            print(f"\nDataset: {dataset_dir} (missing)")
            continue
        inspect_dataset(dataset_dir)


if __name__ == "__main__":
    main()
