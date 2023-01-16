import numpy as np


def noise_original_data(data, noise_level):
    """Add noise to the original data."""
    return data + np.random.normal(scale=noise_level, size=data.shape)
