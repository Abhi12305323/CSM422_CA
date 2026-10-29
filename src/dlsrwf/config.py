import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

_ALLOWED = {
    "experiment": "name output_dir",
    "data": "path validation_fraction split_strategy seed scale",
    "model": (
        "variant filters kernel_sizes dropout l2 batch_norm output_channels "
        "output_activation weights"
    ),
    "training": (
        "batch_size epochs learning_rate loss early_stopping_patience "
        "reduce_lr_patience minimum_learning_rate seed"
    ),
    "evaluation": "context_steps horizon test_path",
}


def _section(document: Mapping[str, Any], name: str) -> dict[str, Any]:
    values = document.get(name, {})
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a YAML mapping.")
    unknown = sorted(set(values) - set(_ALLOWED[name].split()))
    if unknown:
        raise ValueError(f"Unknown {name} option(s): {', '.join(unknown)}.")
    return dict(values)


def _path(base: Path, value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise TypeError(f"{name} must be a non-empty path.")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Configuration must use a .yaml or .yml extension.")
    if not source.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {source}")

    raw = source.read_bytes()
    try:
        document = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid YAML in {source}: {exc}") from exc
    if not isinstance(document, Mapping):
        raise TypeError("configuration must be a YAML mapping.")

    unknown = sorted(set(document) - {"schema_version", *_ALLOWED})
    if unknown:
        raise ValueError(f"Unknown top-level option(s): {', '.join(unknown)}.")
    version = document.get("schema_version", 1)
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"Unsupported schema_version {version!r}; expected 1.")

    sections = {name: _section(document, name) for name in _ALLOWED}
    experiment = sections["experiment"]
    if not {"name", "output_dir"} <= experiment.keys():
        raise ValueError("experiment.name and experiment.output_dir are required.")
    if "path" not in sections["data"]:
        raise ValueError("data.path is required.")

    base = source.parent
    data = {
        "validation_fraction": 0.1,
        "split_strategy": "chronological",
        "seed": 42,
        "scale": True,
        **sections["data"],
    }
    data["path"] = _path(base, data["path"], "data.path")
    model = {
        "variant": "causal",
        "filters": [64, 32, 32],
        "kernel_sizes": [5, 3, 1],
        "dropout": 0.0,
        "l2": 0.0,
        "batch_norm": False,
        "output_channels": 1,
        "output_activation": "sigmoid",
        **sections["model"],
    }
    weights_value = model.pop("weights", None)
    weights = (
        None if weights_value is None else _path(base, weights_value, "model.weights")
    )
    training = {
        "batch_size": 4,
        "epochs": 20,
        "learning_rate": 0.001,
        "loss": "mse",
        "early_stopping_patience": 5,
        "reduce_lr_patience": 3,
        "minimum_learning_rate": 0.000001,
        "seed": 42,
        **sections["training"],
    }
    evaluation = {"context_steps": 12, "horizon": 12, **sections["evaluation"]}
    test_value = evaluation.get("test_path")
    evaluation["test_path"] = (
        None if test_value is None else _path(base, test_value, "evaluation.test_path")
    )
    return {
        "source_path": source,
        "fingerprint": hashlib.sha256(raw).hexdigest(),
        "name": experiment["name"],
        "output_dir": _path(base, experiment["output_dir"], "experiment.output_dir"),
        "data": data,
        "model": model,
        "weights": weights,
        "training": training,
        "evaluation": evaluation,
    }
