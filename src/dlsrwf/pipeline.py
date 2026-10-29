"""Data preparation, training, prediction, and benchmarking."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from dlsrwf.config import load_config
from dlsrwf.data import load_sequences, prepare_dataset, scale_pixels, split_sequences


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    text = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=str) + "\n"
    )
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def _path(value: str | Path | None) -> Path | None:
    return None if value is None else Path(value).expanduser().resolve()


def _space(config: dict[str, Any]) -> dict[str, str]:
    warning = "Rendered-image errors are not physical weather-variable units."
    if config["data"]["scale"]:
        return {
            "name": "normalized_legacy_grayscale_pixel",
            "unit": "unitless",
            "range": "[0, 1] under divide-by-255 scaling",
            "warning": warning,
        }
    return {
        "name": "legacy_grayscale_pixel",
        "unit": "pixel intensity",
        "range": "normally [0, 255]",
        "warning": warning,
    }


def _checkpoint_exists(path: Path) -> bool:
    return path.is_file() or Path(f"{path}.index").is_file()


def _checkpoint_hashes(path: Path) -> dict[str, str]:
    files = [path] if path.is_file() else sorted(path.parent.glob(f"{path.name}.*"))
    if not files:
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    return {file.name: sha256_file(file) for file in files}


def _weights(
    config: dict[str, Any], override: str | Path | None, required: bool = False
) -> Path | None:
    path = _path(override) or config["weights"]
    if path is None:
        if required:
            raise ValueError("Set model.weights in YAML or WEIGHTS_PATH in the script.")
        return None
    if not _checkpoint_exists(path):
        raise FileNotFoundError(f"Model checkpoint does not exist: {path}")
    return path


def _evaluation_data(
    config: dict[str, Any],
) -> tuple[Path, str, np.ndarray]:
    test_path = config["evaluation"]["test_path"]
    if test_path is not None:
        sequences = load_sequences(test_path)
        if config["data"]["scale"]:
            sequences = scale_pixels(sequences)
        return test_path, "test", np.asarray(sequences, dtype=np.float32)

    source = load_sequences(config["data"]["path"])
    _, validation = split_sequences(
        source,
        config["data"]["validation_fraction"],
        config["data"]["split_strategy"],
        config["data"]["seed"],
        config["data"]["scale"],
    )
    return (
        config["data"]["path"],
        "validation",
        np.asarray(validation, dtype=np.float32),
    )


def build_configured_model(
    config: dict[str, Any], sequences: np.ndarray, weights: Path | None = None
) -> Any:
    if sequences.ndim != 5:
        raise ValueError("Model input must have shape (N,T,H,W,C).")
    if sequences.shape[-1] != config["model"]["output_channels"]:
        raise ValueError("Data and model output channels do not match.")

    from dlsrwf.models import build_model

    shape = (None, *(int(size) for size in sequences.shape[2:]))
    model = build_model(config["model"], shape)
    if weights is not None:
        try:
            status = model.load_weights(str(weights))
            if hasattr(status, "assert_nontrivial_match"):
                status.assert_nontrivial_match()
                status.assert_existing_objects_matched()
                status.expect_partial()
        except (AssertionError, KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"Checkpoint is incompatible with the model: {weights}"
            ) from exc
    return model


def _predict(model: Any, batch: np.ndarray) -> np.ndarray:
    try:
        output = model.predict(batch, verbose=0)
    except TypeError:
        output = model.predict(batch)
    return np.asarray(output, dtype=np.float32)


def autoregressive_forecast(
    model: Any, context: np.ndarray, horizon: int
) -> np.ndarray:
    rolling = np.asarray(context, dtype=np.float32)
    if rolling.ndim != 4 or len(rolling) < 1:
        raise ValueError("Context must have shape (T,H,W,C).")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")

    predictions = []
    for _ in range(horizon):
        output = _predict(model, rolling[np.newaxis])
        if output.ndim != 5 or output.shape[0] != 1:
            raise ValueError("Model output must have shape (1,T,H,W,C).")
        frame = output[0, -1]
        if frame.shape != rolling.shape[1:]:
            raise ValueError("Predicted frame shape does not match the context.")
        predictions.append(frame)
        rolling = np.concatenate((rolling[1:], frame[np.newaxis]))
    return np.stack(predictions)


def persistence_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    history = np.asarray(context)
    if history.ndim != 4 or len(history) < 1:
        raise ValueError("Context must have shape (T,H,W,C).")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer.")
    return np.repeat(history[-1:], horizon, axis=0)


def metric_values(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    truth = np.asarray(observed, dtype=np.float64)
    forecast = np.asarray(predicted, dtype=np.float64)
    if truth.shape != forecast.shape:
        raise ValueError("Observed and predicted arrays must have the same shape.")
    mask = np.isfinite(truth) & np.isfinite(forecast)
    if not mask.any():
        raise ValueError("No finite values are available for evaluation.")
    error = forecast[mask] - truth[mask]
    mse = float(np.mean(error**2))
    return {
        "mae": float(np.mean(np.abs(error))),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "bias": float(np.mean(error)),
    }


def evaluate_forecasts(
    sequences: np.ndarray,
    context_steps: int,
    horizon: int,
    forecast: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Any]:
    data = np.asarray(sequences)
    if data.ndim != 5 or data.shape[1] < context_steps + horizon:
        raise ValueError("Evaluation data does not contain the requested window.")
    observed, predicted = [], []
    for sequence in data:
        context = sequence[:context_steps]
        truth = sequence[context_steps : context_steps + horizon]
        estimate = np.asarray(forecast(context, horizon), dtype=np.float32)
        if estimate.shape != truth.shape:
            raise ValueError("Forecast shape does not match the observation.")
        observed.append(truth)
        predicted.append(estimate)
    truth, estimate = np.stack(observed), np.stack(predicted)
    return {
        "samples": len(data),
        "context_steps": context_steps,
        "horizon": horizon,
        "overall": metric_values(truth, estimate),
        "by_lead": [
            {"lead": lead + 1, **metric_values(truth[:, lead], estimate[:, lead])}
            for lead in range(horizon)
        ],
    }


def _fit_model(
    model: Any,
    training: dict[str, Any],
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    run_dir: Path,
    resume: bool,
    verbose: int,
) -> tuple[Any, Path, Path]:
    import tensorflow as tf

    run_dir.mkdir(parents=True, exist_ok=True)
    best = run_dir / "best.weights.h5"
    final = run_dir / "final.weights.h5"
    tf.keras.utils.set_random_seed(training["seed"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(training["learning_rate"]),
        loss=training["loss"],
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    if resume:
        if not best.is_file():
            raise FileNotFoundError(f"Cannot resume: {best} does not exist.")
        model.load_weights(best)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(best), monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=training["early_stopping_patience"]
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=training["reduce_lr_patience"],
            min_lr=training["minimum_learning_rate"],
        ),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv"), append=resume),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    x_train, y_train, x_validation, y_validation = arrays
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        batch_size=training["batch_size"],
        epochs=training["epochs"],
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )
    model.save_weights(final)
    if best.is_file():
        model.load_weights(best)
    write_json(
        run_dir / "run.json",
        {"training": training, "history": history.history},
    )
    return history, best, final


def prepare_data(
    config_path: str | Path,
    *,
    output_path: str | Path | None = None,
    arrays_path: str | Path | None = None,
    calculate_checksum: bool = True,
) -> Path:
    config = load_config(config_path)
    source, train, validation, x_train, y_train, x_val, y_val = prepare_dataset(
        config["data"]
    )
    arrays = _path(arrays_path)
    if arrays is not None:
        if arrays.suffix.lower() != ".npz":
            raise ValueError("ARRAYS_PATH must end in .npz.")
        arrays.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            arrays,
            x_train=x_train,
            y_train=y_train,
            x_validation=x_val,
            y_validation=y_val,
        )
    shapes = {
        "train_shape": list(train.shape),
        "validation_shape": list(validation.shape),
        "x_train_shape": list(x_train.shape),
        "y_train_shape": list(y_train.shape),
        "x_validation_shape": list(x_val.shape),
        "y_validation_shape": list(y_val.shape),
        "arrays": None if arrays is None else str(arrays),
    }
    source_path = config["data"]["path"]
    payload = {
        "schema_version": 1,
        "experiment": config["name"],
        "config": str(config["source_path"]),
        "config_fingerprint": config["fingerprint"],
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path) if calculate_checksum else None,
            "shape": list(source.shape),
            "dtype": str(source.dtype),
            "minimum": float(source.min()),
            "maximum": float(source.max()),
        },
        "preparation": {
            "split_strategy": config["data"]["split_strategy"],
            "validation_fraction": config["data"]["validation_fraction"],
            "seed": config["data"]["seed"],
            "scaled": config["data"]["scale"],
            "space": _space(config),
            **shapes,
        },
    }
    target = _path(output_path) or config["output_dir"] / "data_manifest.json"
    result = write_json(target, payload)
    print(f"Wrote {result}")
    if arrays is not None:
        print(f"Wrote {arrays}")
    return result


def train_model(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    verbose: int = 1,
) -> Path:
    if resume and weights_path is not None:
        raise ValueError("WEIGHTS_PATH and RESUME cannot be used together.")
    if verbose not in (0, 1, 2):
        raise ValueError("VERBOSE must be 0, 1, or 2.")
    config = load_config(config_path)
    _, _, _, x_train, y_train, x_val, y_val = prepare_dataset(config["data"])
    initial = None if resume else _weights(config, weights_path)
    model = build_configured_model(config, x_train, initial)
    run_dir = _path(output_dir) or config["output_dir"] / "training"
    write_json(run_dir / "resolved_config.json", config)
    history, best, final = _fit_model(
        model,
        config["training"],
        (x_train, y_train, x_val, y_val),
        run_dir,
        resume,
        verbose,
    )
    last_metrics = {
        name: float(values[-1]) for name, values in history.history.items() if values
    }
    summary = {
        "schema_version": 1,
        "experiment": config["name"],
        "config": str(config["source_path"]),
        "config_fingerprint": config["fingerprint"],
        "data": {
            "path": str(config["data"]["path"]),
            "sha256": sha256_file(config["data"]["path"]),
            "space": _space(config),
            "x_train_shape": list(x_train.shape),
            "y_train_shape": list(y_train.shape),
            "x_validation_shape": list(x_val.shape),
            "y_validation_shape": list(y_val.shape),
        },
        "model": {
            "variant": config["model"]["variant"],
            "initial_checkpoint": None if initial is None else str(initial),
            "initial_checkpoint_sha256": (
                None if initial is None else _checkpoint_hashes(initial)
            ),
        },
        "training": {
            "resumed": resume,
            "epochs_completed": len(history.epoch),
            "last_epoch_metrics": last_metrics,
            "best_weights": str(best),
            "best_weights_sha256": _checkpoint_hashes(best),
            "final_weights": str(final),
            "final_weights_sha256": _checkpoint_hashes(final),
        },
    }
    result = write_json(run_dir / "training_summary.json", summary)
    print(f"Wrote {result}")
    print(f"Wrote {final}")
    return result


def predict(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    sample_index: int = 0,
    output_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    weights = _weights(config, weights_path, required=True)
    assert weights is not None
    data_path, split, sequences = _evaluation_data(config)
    context_steps = config["evaluation"]["context_steps"]
    horizon = config["evaluation"]["horizon"]
    if sequences.shape[1] < context_steps + horizon:
        raise ValueError("Evaluation data is shorter than context plus horizon.")
    if sample_index < 0 or sample_index >= len(sequences):
        raise IndexError(f"SAMPLE_INDEX must be between 0 and {len(sequences) - 1}.")
    model = build_configured_model(config, sequences, weights)
    sequence = sequences[sample_index]
    context = sequence[:context_steps]
    observation = sequence[context_steps : context_steps + horizon]
    forecast = autoregressive_forecast(model, context, horizon)
    target = _path(output_path) or config["output_dir"] / "prediction.npz"
    if target.suffix.lower() != ".npz":
        raise ValueError("OUTPUT_PATH must end in .npz.")
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target, context=context, prediction=forecast, observation=observation
    )
    write_json(
        target.with_suffix(".json"),
        {
            "schema_version": 1,
            "experiment": config["name"],
            "config": str(config["source_path"]),
            "config_fingerprint": config["fingerprint"],
            "dataset": {
                "path": str(data_path),
                "sha256": sha256_file(data_path),
                "split": split,
                "sample_index": sample_index,
            },
            "model": {
                "variant": config["model"]["variant"],
                "checkpoint": str(weights),
                "checkpoint_sha256": _checkpoint_hashes(weights),
            },
            "forecast": {
                "mode": "autoregressive",
                "context_steps": context_steps,
                "horizon": horizon,
                "future_observations_available_to_forecaster": False,
                "space": _space(config),
                "archive": str(target),
            },
        },
    )
    print(f"Wrote {target}")
    print(f"Wrote {target.with_suffix('.json')}")
    return target


def benchmark(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    persistence_only: bool = False,
    max_samples: int | None = None,
    output_path: str | Path | None = None,
) -> Path:
    config = load_config(config_path)
    data_path, split, sequences = _evaluation_data(config)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("MAX_SAMPLES must be positive.")
        sequences = sequences[:max_samples]
    context_steps = config["evaluation"]["context_steps"]
    horizon = config["evaluation"]["horizon"]
    persistence = evaluate_forecasts(
        sequences, context_steps, horizon, persistence_forecast
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": config["name"],
        "config": str(config["source_path"]),
        "config_fingerprint": config["fingerprint"],
        "dataset": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "split": split,
            "split_strategy": (
                config["data"]["split_strategy"] if split == "validation" else None
            ),
            "validation_fraction": (
                config["data"]["validation_fraction"] if split == "validation" else None
            ),
            "seed": config["data"]["seed"] if split == "validation" else None,
            "samples": len(sequences),
            "shape": list(sequences.shape),
        },
        "evaluation": {
            "context_steps": context_steps,
            "horizon": horizon,
            "space": _space(config),
        },
        "protocol": {
            "forecast_mode": "autoregressive",
            "future_observations_available_to_forecaster": False,
            "comparison_scope": "Internal V1 comparison in legacy image space.",
        },
        "systems": {"persistence": persistence},
    }
    weights = None if persistence_only else _weights(config, weights_path)
    if weights is None:
        result["model"] = {
            "status": "not_run",
            "reason": "Persistence-only run or no checkpoint supplied.",
        }
    else:
        model = build_configured_model(config, sequences, weights)
        report = evaluate_forecasts(
            sequences,
            context_steps,
            horizon,
            lambda context, steps: autoregressive_forecast(model, context, steps),
        )
        result["systems"]["convlstm"] = report
        result["model"] = {
            "status": "evaluated",
            "variant": config["model"]["variant"],
            "checkpoint": str(weights),
            "checkpoint_sha256": _checkpoint_hashes(weights),
        }
        baseline_mse = persistence["overall"]["mse"]
        result["comparison"] = {
            "mse_skill_vs_persistence": (
                None
                if baseline_mse == 0
                else 1 - report["overall"]["mse"] / baseline_mse
            ),
            "positive_skill_is_better": True,
        }
    target = _path(output_path) or config["output_dir"] / "benchmark.json"
    result_path = write_json(target, result)
    print(f"Wrote {result_path}")
    return result_path
