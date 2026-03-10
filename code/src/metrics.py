import numpy as np
from typing import Callable, List, Tuple


def estimate_pfa(
    detector_fn: Callable[[np.ndarray], Tuple[int, float]],
    noise_generator_fn: Callable[[], np.ndarray],
    num_trials: int
) -> Tuple[float, np.ndarray]:
    decisions = np.zeros(num_trials, dtype=int)
    tmc_values = np.zeros(num_trials, dtype=float)

    for m in range(num_trials):
        x = noise_generator_fn()
        decision, tmc = detector_fn(x)
        decisions[m] = decision
        tmc_values[m] = tmc

    pfa_estimate = np.mean(decisions)
    return float(pfa_estimate), tmc_values


def estimate_pd(
    detector_fn: Callable[[np.ndarray], Tuple[int, float]],
    signal_plus_noise_generator_fn: Callable[[], np.ndarray],
    num_trials: int
) -> Tuple[float, np.ndarray]:
    decisions = np.zeros(num_trials, dtype=int)
    tmc_values = np.zeros(num_trials, dtype=float)

    for m in range(num_trials):
        x = signal_plus_noise_generator_fn()
        decision, tmc = detector_fn(x)
        decisions[m] = decision
        tmc_values[m] = tmc

    pd_estimate = np.mean(decisions)
    return float(pd_estimate), tmc_values


def sweep_pd_vs_snr(
    detector_fn: Callable[[np.ndarray], Tuple[int, float]],
    snr_db_values: List[float],
    signal_generator_fn: Callable[[float], np.ndarray],
    noise_generator_fn: Callable[[], np.ndarray],
    num_samples: int,
    num_trials: int
) -> Tuple[np.ndarray, np.ndarray]:
    snr_db_array = np.array(snr_db_values)
    pd_array = np.zeros(len(snr_db_values))

    for i, snr_db in enumerate(snr_db_values):
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = snr_linear 

        def h1_generator():
            signal = signal_generator_fn(signal_power)
            noise = noise_generator_fn()
            return signal + noise

        pd_array[i], _ = estimate_pd(detector_fn, h1_generator, num_trials)

    return snr_db_array, pd_array


def compute_empirical_pfa_vs_threshold(
    tmc_h0_values: np.ndarray,
    threshold_range: np.ndarray
) -> np.ndarray:
    pfa_curve = np.array([
        np.mean(tmc_h0_values > lam) for lam in threshold_range
    ])
    return pfa_curve


def compute_theoretical_pfa_vs_threshold(threshold_range: np.ndarray) -> np.ndarray:
    return 1.0 / (threshold_range + 1.0)


def summarize_results(snr_db: np.ndarray, pd: np.ndarray, pfa: float, threshold: float) -> str:
    lines = [
        "=" * 60,
        "  Multi-Cycle Cyclostationary Detector — Simulation Results",
        "=" * 60,
        f"  Detection Threshold (lambda):   {threshold:.4f}",
        f"  Empirical Pfa (H0 trials):      {pfa:.4f}",
        f"  Theoretical Pfa:                {1.0 / (threshold + 1.0):.4f}",
        "-" * 60,
        f"  {'SNR (dB)':>10}   {'Pd':>8}",
        "-" * 60,
    ]
    for s, p in zip(snr_db, pd):
        lines.append(f"  {s:>10.1f}   {p:>8.4f}")
    lines.append("=" * 60)
    return "\n".join(lines)
