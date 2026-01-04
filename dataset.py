import os

from datasets import load_dataset
from tqdm import tqdm


DATASET_NAMES = [
    "wanglab/kegg",
    "wanglab/variant_effect_coding",
    "wanglab/variant_effect_non_snv",
]


def dataset_dir_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "___")


def export_dataset(dataset_name: str, cache_dir: str, output_root: str) -> None:
    ds = load_dataset(dataset_name, cache_dir=cache_dir)
    output_dir = os.path.join(output_root, dataset_dir_name(dataset_name))
    os.makedirs(output_dir, exist_ok=True)

    arrow_dir = os.path.join(output_dir, "arrow")
    ds.save_to_disk(arrow_dir)

    for split_name in tqdm(list(ds.keys()), desc=f"CSV {dataset_name}"):
        split = ds[split_name]
        split_path = os.path.join(output_dir, f"{split_name}.csv")
        split.to_csv(split_path)
        print(f"Wrote {split_name} to {split_path}")


def main() -> None:
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(project_dir, "dataset")
    os.makedirs(dataset_root, exist_ok=True)

    for dataset_name in tqdm(DATASET_NAMES, desc="Datasets"):
        print(f"Loading {dataset_name}")
        export_dataset(dataset_name, cache_dir=dataset_root, output_root=dataset_root)


if __name__ == "__main__":
    main()
