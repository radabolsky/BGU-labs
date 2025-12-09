import numpy as np
from struct import unpack
from pathlib import Path


def load_signal(file_path):
    with open(file_path, "br") as f:
        signal = np.array(unpack("10000H", f.read()))
    return signal


def load_all_signals(data_dir):
    data_dir = Path(data_dir)
    signals = []
    for signal_path in sorted(data_dir.glob("d*")):
        if signal_path.suffix in [".xls", ".tmp"] or signal_path.is_dir():
            continue
        try:
            signal = load_signal(signal_path)
            signals.append(signal)
        except Exception as e:
            print(f"Error loading {signal_path}: {e}")
            continue
    return np.array(signals)
