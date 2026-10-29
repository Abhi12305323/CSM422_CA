"""Loading and splitting the legacy NumPy weather sequences."""

from pathlib import Path
from typing import Any

import numpy as np


def as_sequences(data: Any) -> np.ndarray:
    array = np.asanyarray(data)
    if array.ndim not in (4, 5):
        raise ValueError("Expected data shaped (N,T,H,W) or (N,T,H,W,C).")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("Sequence data must contain real numbers.")
    if array.shape[0] < 1 or array.shape[1] < 2 or 0 in array.shape:
        raise ValueError("Sequence data must contain samples and at least two frames.")
    if not np.isfinite(array).all():
        raise ValueError("Sequence data contains NaN or infinite values.")
    return array[..., np.newaxis] if array.ndim == 4 else array


def load_sequences(path: str | Path) -> np.ndarray:
    source = Path(path)
    if source.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file: {source}")
    if not source.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {source}")
    try:
        return as_sequences(np.load(source, mmap_mode="r", allow_pickle=False))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load {source}: {exc}") from exc


def scale_pixels(data: Any) -> np.ndarray:
    array = as_sequences(data)
    if float(array.min()) < 0 or float(array.max()) > 255:
        raise ValueError("Pixel values must be within [0, 255].")
    return array.astype(np.float32) / 255.0


def split_sequences(
    data: Any,
    validation_fraction: float = 0.1,
    strategy: str = "chronological",
    seed: int = 42,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    array = as_sequences(data)
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if strategy not in ("chronological", "random"):
        raise ValueError("split_strategy must be chronological or random.")
    count = int(np.ceil(len(array) * validation_fraction))
    if count >= len(array):
        raise ValueError("The split leaves no training samples.")
    if strategy == "chronological":
        train, validation = array[:-count], array[-count:]
    else:
        indices = np.random.default_rng(seed).permutation(len(array))
        train, validation = array[indices[:-count]], array[indices[-count:]]
    if scale:
        train, validation = scale_pixels(train), scale_pixels(validation)
    return train, validation


def shifted_pairs(data: Any) -> tuple[np.ndarray, np.ndarray]:
    array = as_sequences(data)
    return array[:, :-1], array[:, 1:]


def prepare_dataset(settings: dict[str, Any]) -> tuple[np.ndarray, ...]:
    source = load_sequences(settings["path"])
    train, validation = split_sequences(
        source,
        settings["validation_fraction"],
        settings["split_strategy"],
        settings["seed"],
        settings["scale"],
    )
    x_train, y_train = shifted_pairs(train)
    x_validation, y_validation = shifted_pairs(validation)
    return source, train, validation, x_train, y_train, x_validation, y_validation
