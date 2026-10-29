"""Small plotting helper used by the example notebook."""

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure


def comparison_plot(
    observed: Any,
    predicted: Any,
    output_path: str | Path,
    *,
    cmap: str = "viridis",
    max_leads: int = 12,
) -> Path:
    truth = np.asarray(observed)
    forecast = np.asarray(predicted)
    if truth.ndim == 4 and truth.shape[-1] == 1:
        truth, forecast = truth[..., 0], forecast[..., 0]
    if truth.ndim != 3 or truth.shape != forecast.shape:
        raise ValueError("Observed and predicted frames must have the same 3-D shape.")
    if max_leads < 1:
        raise ValueError("max_leads must be positive.")

    leads = min(len(truth), max_leads)
    figure = Figure(figsize=(2.2 * leads, 5))
    axes = figure.subplots(2, leads, squeeze=False)
    for lead in range(leads):
        axes[0, lead].imshow(truth[lead], cmap=cmap)
        axes[1, lead].imshow(forecast[lead], cmap=cmap)
        axes[0, lead].set_title(f"Observed +{lead + 1} h")
        axes[1, lead].set_title(f"Forecast +{lead + 1} h")
        axes[0, lead].axis("off")
        axes[1, lead].axis("off")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=150, bbox_inches="tight")
    figure.clear()
    return target
