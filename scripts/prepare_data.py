from pathlib import Path

from dlsrwf.pipeline import prepare_data

ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "sample_precipitation.yaml"
OUTPUT_PATH = None
ARRAYS_PATH = None
CALCULATE_CHECKSUM = True


if __name__ == "__main__":
    prepare_data(
        CONFIG_PATH,
        output_path=OUTPUT_PATH,
        arrays_path=ARRAYS_PATH,
        calculate_checksum=CALCULATE_CHECKSUM,
    )
