# Data directory

The complete historical datasets are intentionally not stored in Git. This
directory follows the layout below:

```text
data/
├── raw/          # source downloads or exported frames
├── interim/      # cropped, resized, or otherwise intermediate files
├── processed/    # model-ready local arrays
├── local/        # private, machine-specific data
└── sample/       # small versioned data used for examples and smoke tests
```

The first four directories are ignored. Create them locally as needed; do not
force-add their contents.

## Bundled samples

| File | Shape | Dtype | Range | SHA-256 |
|---|---:|---|---:|---|
| `sample/precipitation_2022.npy` | `(31, 24, 74, 104)` | `int16` | `76`–`255` | `887d3d3d6a101e1931c2d9cdea9ba64a084dad523b2b91d3f128e76e24d8f29c` |
| `sample/precipitation_2022_resized.npy` | `(31, 24, 51, 72)` | `int16` | `76`–`255` | `3eee8cc80cf178fd21e13be3e506ed08d618054dcc4413e19e574a7371c4b87a` |

They contain legacy grayscale visualization pixels, not precipitation in
millimetres. The resized variant matches the preprocessing recorded for the
selected precipitation checkpoint. Neither array contains timestamps,
coordinates, units, or source-file identifiers, so exact sample dates cannot be
reconstructed from the files alone. They are suitable for inspecting the data
contract and exercising the pipeline; they are not training datasets or
standalone scientific benchmarks.

Load the sample through the same validation used by the scripts:

```python
from dlsrwf.data import load_sequences

sequences = load_sequences("data/sample/precipitation_2022.npy")
```

From the repository root, run `python scripts/prepare_data.py` to validate the
sample, make the deterministic split, and write its manifest under `outputs/`.

See [the data documentation](../docs/data.md) for provenance, historical array
shapes, preprocessing limitations, and split rules.
