# quantization-metrics

Numpy-only metrics for comparing original vs quantized ML signals.

## Install

pip install -e .

## Usage

from quantization_metrics import mse_from_signal, snr_db

original = ...
quantized = ...
print(snr_db(original, quantized))