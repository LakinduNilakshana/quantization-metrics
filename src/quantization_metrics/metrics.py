import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def _prepare_signals(
    original: FloatArray,
    quantized: FloatArray,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate shape match and return flattened float64 views."""
    if original.shape != quantized.shape:
        raise ValueError("Original and quantized signals must have the same shape.")
    return (
        np.asarray(original, dtype=np.float64).ravel(),
        np.asarray(quantized, dtype=np.float64).ravel(),
    )


def _mse(orig: NDArray[np.float64], quant: NDArray[np.float64]) -> float:
    return float(np.mean((orig - quant) ** 2))


def _signal_power(orig: NDArray[np.float64]) -> float:
    return float(np.mean(orig**2))


def mse_from_signal(
    original: FloatArray,
    quantized: FloatArray,
) -> float:
    """Mean squared error between original and quantized signals."""
    orig, quant = _prepare_signals(original, quantized)
    return _mse(orig, quant)


def nmse(
    original: FloatArray,
    quantized: FloatArray,
) -> float:
    """Normalized mean squared error: MSE / mean(original²)."""
    orig, quant = _prepare_signals(original, quantized)

    if orig.size == 0:
        return float("nan")

    power = _signal_power(orig)
    mse = _mse(orig, quant)

    if power == 0:
        return float("inf") if mse > 0 else float("nan")

    return mse / power


def snr_db(
    original: FloatArray,
    quantized: FloatArray,
) -> float:
    """Signal-to-noise ratio in decibels."""
    orig, quant = _prepare_signals(original, quantized)

    if orig.size == 0:
        return float("nan")

    power = _signal_power(orig)
    mse = _mse(orig, quant)

    if mse == 0:
        return float("inf")
    if power == 0:
        return float("-inf")

    return float(10 * np.log10(power / mse))


def correlation_coeff(
    original: FloatArray,
    quantized: FloatArray,
) -> float:
    """Pearson correlation coefficient between original and quantized."""
    orig, quant = _prepare_signals(original, quantized)

    if orig.size < 2:
        return float("nan")

    if np.std(orig) == 0 or np.std(quant) == 0:
        return 1.0 if np.array_equal(orig, quant) else float("nan")

    return float(np.corrcoef(orig, quant)[0, 1])
