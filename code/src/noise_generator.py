#Generates complex Gaussian noise samples with optional variance uncertainty.
import numpy as np
def generate_complex_gaussian_noise(
    num_samples: int,
    variance: float = 1.0,
    rng: np.random.Generator = None
) -> np.ndarray:
    
    """Here,
    num_samples :Number of noise samples to generate.
    variance :Total noise power E[|w[n]|^2] = sigma^2.
    rng:Random generator for reproducibility.
    """

    if rng is None:
        rng = np.random.default_rng()

    std_per_component = np.sqrt(variance / 2.0)
    real_part = rng.normal(0.0, std_per_component, size=num_samples)
    imag_part = rng.normal(0.0, std_per_component, size=num_samples)
    return real_part + 1j * imag_part

def sample_uncertain_noise_variance(
    nominal_variance: float,
    uncertainty_db: float,
    rng: np.random.Generator = None
) -> float:
    
    """Here,
    nominal_variance :The assumed noise variance sigma^2.
    uncertainty_db : Noise uncertainty level in dB (e.g., 1.0 dB).
    """
    if rng is None:
        rng = np.random.default_rng()

    epsilon = 10 ** (uncertainty_db / 10.0)
    lower = nominal_variance / epsilon
    upper = nominal_variance * epsilon
    return rng.uniform(lower, upper)


def generate_noise_with_uncertainty(
    num_samples: int,
    nominal_variance: float,
    uncertainty_db: float = 0.0,
    rng: np.random.Generator = None
) -> np.ndarray:

    if rng is None:
        rng = np.random.default_rng()

    if uncertainty_db > 0:
        actual_variance = sample_uncertain_noise_variance(
            nominal_variance, uncertainty_db, rng
        )
    else:
        actual_variance = nominal_variance

    return generate_complex_gaussian_noise(num_samples, actual_variance, rng)
