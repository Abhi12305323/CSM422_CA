from pathlib import Path

from dlsrwf.pipeline import train_model

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "sample_precipitation.yaml"
WEIGHTS_PATH = None
OUTPUT_DIR = None
RESUME = False
VERBOSE = 1


if __name__ == "__main__":
    train_model(
        CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        output_dir=OUTPUT_DIR,
        resume=RESUME,
        verbose=VERBOSE,
    )
