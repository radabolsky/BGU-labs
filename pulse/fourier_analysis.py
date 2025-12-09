import numpy as np
from scipy.fft import fft, fftfreq
from pulse.helpers import preprocess_signal


def find_main_frequency(signal, sampling_rate=100.0):
    n = len(signal)
    fft_values = fft(signal)
    fft_magnitude = np.abs(fft_values)
    frequencies = fftfreq(n, 1 / sampling_rate)

    positive_freq_idx = frequencies > 0
    positive_frequencies = frequencies[positive_freq_idx]
    positive_magnitude = fft_magnitude[positive_freq_idx]

    main_freq_idx = np.argmax(positive_magnitude)
    main_frequency = positive_frequencies[main_freq_idx]

    return main_frequency


def compute_fourier_frequencies(signals, sampling_rate=100.0):
    frequencies = []
    for signal in signals:
        freq = find_main_frequency(preprocess_signal(signal), sampling_rate)
        frequencies.append(freq)
    return np.array(frequencies)
