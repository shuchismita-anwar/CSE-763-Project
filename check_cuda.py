import torch


def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda build: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"cudnn enabled: {torch.backends.cudnn.enabled}")
    print(f"device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print(f"current device: {idx}")
        print(f"name: {torch.cuda.get_device_name(idx)}")
        print(f"capability: {props.major}.{props.minor}")
        print(f"total memory (GB): {props.total_memory / (1024 ** 3):.2f}")


if __name__ == "__main__":
    main()
