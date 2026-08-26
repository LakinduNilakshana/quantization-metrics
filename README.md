# quantization-metrics

Numpy-only metrics for comparing original vs quantized ML signals.

## Install

```bash
pip install -e .
```

## Usage

```python
from quantization_metrics import mse_from_signal, snr_db

original = ...
quantized = ...
print(snr_db(original, quantized))
```

## License

MIT — see [LICENSE](LICENSE).
