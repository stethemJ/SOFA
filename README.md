# SOFA
Spherical-harmonic Optimizer for Forward-modeled Antennas

## Setup

```
python3 -m venv env
source env/bin/activate
pip install -e .
```

This installs `antenna` and `forward_model` as importable packages, so any script can do `from forward_model.tools.some_module import SomeClass`.

## Saving plots

Scripts should save plot output through `tools.config.OUTPUT_DIR`, which
resolves to the `outputs/` folder (override with the `SOFA_OUTPUT_DIR` env
var, already set for you when `env/bin/activate` is sourced):

```python
from forward_model.tools.config import OUTPUT_DIR
import matplotlib.pyplot as plt

plt.savefig(OUTPUT_DIR / "my_plot.png")
```
