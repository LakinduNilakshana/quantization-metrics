# quantization-metrics

Numpy-only metrics for comparing **original** vs **quantized** ML signals (weights or activations).

The API is small on purpose: four functions, no PyTorch/TF dependency. Tests, CI, and a PyPI release come next.

## Metrics

| Function | What it measures | Better when |
|----------|------------------|-------------|
| `mse_from_signal` | Mean squared error | Lower |
| `nmse` | MSE / mean(original²) | Lower |
| `snr_db` | Signal-to-noise ratio (dB) | Higher |
| `correlation_coeff` | Pearson correlation | Closer to 1.0 |

## Install

```bash
pip install -e .
```

Requires Python 3.11+ and numpy.

## Usage

```python
import numpy as np
from quantization_metrics import mse_from_signal, nmse, snr_db, correlation_coeff

rng = np.random.default_rng(0)
original = rng.normal(size=1000)
quantized = np.round(original * 8) / 8  # stand-in for a coarse quantizer

print(mse_from_signal(original, quantized))
print(nmse(original, quantized))
print(snr_db(original, quantized))
print(correlation_coeff(original, quantized))
```

Shapes must match. Arrays are flattened internally so tensors of any rank work.

## Layout

```
src/quantization_metrics/
  __init__.py      # public API
  metrics.py       # MSE, NMSE, SNR, correlation
```

## License

MIT — see [LICENSE](LICENSE).
