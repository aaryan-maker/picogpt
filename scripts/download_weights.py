import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pico import weights


def main() -> None:
    print("downloading (cached after first run)...")
    path = weights.download()
    print(f"weights at: {path}\n")

    st = weights.load(path)
    print(weights.summarize(st))


if __name__ == "__main__":
    main()
