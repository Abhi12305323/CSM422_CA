from pathlib import Path

from dlsrwf.pipeline import benchmark

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "sample_precipitation.yaml"
WEIGHTS_PATH = None
PERSISTENCE_ONLY = True
MAX_SAMPLES = None
OUTPUT_PATH = None


if __name__ == "__main__":
    benchmark(
        CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        persistence_only=PERSISTENCE_ONLY,
        max_samples=MAX_SAMPLES,
        output_path=OUTPUT_PATH,
    )
