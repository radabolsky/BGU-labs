import numpy as np
from scipy.stats import skew


def preprocess_signal(signal, normalize=False, verbose=False):
    signal_centered = signal - np.mean(signal)
    corrected_signal = signal_centered
    signal_skew = skew(signal_centered)

    if signal_skew < 0:
        corrected_signal = -signal_centered + np.mean(signal)
        if verbose:
            print("Signal reversed")

    if normalize:
        std = np.std(corrected_signal)
        if std > 1e-10:
            corrected_signal = corrected_signal / std
    return corrected_signal
