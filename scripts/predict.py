from pathlib import Path

from dlsrwf.pipeline import predict

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "sample_precipitation.yaml"
WEIGHTS_PATH = (
    ROOT / "outputs" / "sample_precipitation" / "training" / "best.weights.h5"
)
SAMPLE_INDEX = 0
OUTPUT_PATH = None


if __name__ == "__main__":
    predict(
        CONFIG_PATH,
        weights_path=WEIGHTS_PATH,
        sample_index=SAMPLE_INDEX,
        output_path=OUTPUT_PATH,
    )
