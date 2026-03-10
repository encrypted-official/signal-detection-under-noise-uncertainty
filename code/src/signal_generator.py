"""
Generates OFDM signals with cyclic prefix for cognitive radio spectrum sensing simulation.

The OFDM signal exhibits cyclostationary structure due to the cyclic prefix (CP),
which is the key property exploited by the multi-cycle detector.
"""

import numpy as np


def generate_qpsk_symbols(num_symbols: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate random QPSK modulation symbols.

    Each symbol is drawn uniformly from the QPSK constellation:
    {+1+j, +1-j, -1+j, -1-j} / sqrt(2)
    """
    bits = rng.integers(0, 2, size=(num_symbols, 2))
    real_part = (2 * bits[:, 0] - 1).astype(float)
    imag_part = (2 * bits[:, 1] - 1).astype(float)
    return (real_part + 1j * imag_part) / np.sqrt(2)


def generate_ofdm_symbol(
    fft_size: int,
    cp_length: int,
    data_symbols: np.ndarray
) -> np.ndarray:

    # Generate a single OFDM symbol (IFFT + cyclic prefix insertion).
    time_domain = np.fft.ifft(data_symbols, n=fft_size)

    cyclic_prefix = time_domain[-cp_length:]
    return np.concatenate([cyclic_prefix, time_domain])


def generate_ofdm_signal(
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    total_samples: int,
    signal_power: float = 1.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Generate a multi-symbol OFDM signal with specified total sample count.

    The cyclic prefix creates cyclostationary periodicity at cyclic frequencies:
    alpha_k = k / (fft_size + cp_length),  k = {range of all int negative to positive nums}
    """
    if rng is None:
        rng = np.random.default_rng()

    symbol_length = fft_size + cp_length
    total_needed = num_symbols * symbol_length

    signal_parts = []
    for _ in range(num_symbols):
        data = generate_qpsk_symbols(fft_size, rng)
        symbol = generate_ofdm_symbol(fft_size, cp_length, data)
        signal_parts.append(symbol)

    full_signal = np.concatenate(signal_parts)

    if len(full_signal) >= total_samples:
        signal = full_signal[:total_samples]
    else:
        repeats = int(np.ceil(total_samples / len(full_signal)))
        signal = np.tile(full_signal, repeats)[:total_samples]

    current_power = np.mean(np.abs(signal) ** 2)
    if current_power > 0:
        signal = signal * np.sqrt(signal_power / current_power)

    return signal
