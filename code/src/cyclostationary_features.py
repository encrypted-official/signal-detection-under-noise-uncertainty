"""
Computes cyclic autocorrelation (CAC) estimates from received signal samples.

The estimated CAC is the key feature used for OFDM signal detection.
OFDM signals exhibit non-zero CAC at specific cyclic frequencies due to the
cyclic prefix, while noise has approximately zero CAC for all alpha != 0.

Estimated CAC (finite-sample version of theoretical CAC):
    R_hat_x^alpha(tau) = (1/N_tau) * sum_{n=0}^{N_tau-1} x[n] * conj(x[n+tau]) * exp(-j2*pi*alpha*n)

where N_tau = N - tau.

For OFDM signals with FFT size N_FFT and CP length N_CP, non-zero CAC exists at:
    tau   = N_FFT
    alpha_k = k / (N_FFT + N_CP),   k = 0, ±1, ±2, ...
"""

import numpy as np
from typing import List


def estimate_cyclic_autocorrelation(
    x: np.ndarray,
    tau: int,
    alpha: float
) -> complex:
    """
    Estimate the cyclic autocorrelation (CAC) of signal x at lag tau and
    cyclic frequency alpha.

    Formula:
        R_hat_x^alpha(tau) = (1/N_tau) * sum_{n=0}^{N_tau-1}
                              x[n] * conj(x[n + tau]) * exp(-j2*pi*alpha*n)

    Parameters
    ----------
    x : np.ndarray, shape (N,), dtype complex128
        Received signal samples.
    tau : int
        Time lag in samples (typically = FFT size for OFDM detection).
    alpha : float
        Cyclic frequency (normalized). For OFDM: k / (N_FFT + N_CP).
    """
    N = len(x)
    N_tau = N - tau

    if N_tau <= 0:
        raise ValueError(f"Lag tau={tau} must be less than signal length N={N}.")

    n = np.arange(N_tau)
    # Conjugate product at lag tau
    lag_product = x[:N_tau] * np.conj(x[tau:N])
    # Modulate by cyclic frequency
    modulation = np.exp(-1j * 2 * np.pi * alpha * n)
    return np.sum(lag_product * modulation) / N_tau


def compute_ofdm_cyclic_frequencies(
    fft_size: int,
    cp_length: int,
    k_values: List[int]
) -> List[float]:
    symbol_period = fft_size + cp_length
    return [k / symbol_period for k in k_values]


def compute_multi_cycle_numerator(
    x: np.ndarray,
    tau: int,
    cyclic_freqs: List[float]
) -> float:
    N = len(x)
    N_tau = N - tau

    if N_tau <= 0:
        raise ValueError(f"Lag tau={tau} must be less than signal length N={N}.")

    n = np.arange(N_tau, dtype=float)
    K = len(cyclic_freqs)

    # Compute sum of complex exponentials: sum_k exp(-j2*pi*alpha_k*n)
    # Shape: (N_tau,)
    exp_sum = np.zeros(N_tau, dtype=complex)
    for alpha_k in cyclic_freqs:
        exp_sum += np.exp(-1j * 2 * np.pi * alpha_k * n)

    # Compute eta_n = |sum_k exp(-j2*pi*alpha_k*n)| (magnitude of exp_sum)
    # This equals sqrt( (sum cos)^2 + (sum sin)^2 )
    eta_n = np.abs(exp_sum)

    # Avoid division by zero: where eta_n is very small, set weight to 0
    safe_eta = np.where(eta_n > 1e-12, eta_n, 1.0)
    inv_eta = np.where(eta_n > 1e-12, 1.0 / safe_eta, 0.0)

    # Lag product: x[n] * conj(x[n+tau])
    lag_product = x[:N_tau] * np.conj(x[tau:N])

    # Weighted sum: (1/eta_n) * lag_product * exp_sum
    weighted_sum = np.sum(inv_eta * lag_product * exp_sum)

    # Numerator = (1/N_tau) * |weighted_sum|^2
    return (1.0 / N_tau) * np.abs(weighted_sum) ** 2


def compute_reference_energy(
    x: np.ndarray,
    tau_bar: int,
    beta: float
) -> float:
    N = len(x)
    N_tau_bar = N - tau_bar

    if N_tau_bar <= 0:
        raise ValueError(f"Reference lag tau_bar={tau_bar} must be less than N={N}.")

    n = np.arange(N_tau_bar, dtype=float)
    lag_product = x[:N_tau_bar] * np.conj(x[tau_bar:N])
    modulation = np.exp(-1j * 2 * np.pi * beta * n)

    cac_sum = np.sum(lag_product * modulation)
    return (1.0 / N_tau_bar) * np.abs(cac_sum) ** 2