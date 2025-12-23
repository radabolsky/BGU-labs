import numpy as np
from struct import unpack
from pathlib import Path


def load_signal(file_path):
    with open(file_path, "rb") as f:
        signal = np.array(unpack("10000H", f.read()))
    return signal


def load_all_signals(data_dir, return_paths=False):
    """
    Загрузка всех сигналов из директории.
    
    Parameters
    ----------
    data_dir : str or Path
        Директория с сигналами
    return_paths : bool
        Если True, возвращает также список путей к файлам
    
    Returns
    -------
    signals : np.ndarray
        Массив сигналов
    paths : list of Path (optional)
        Список путей к файлам (если return_paths=True)
    """
    data_dir = Path(data_dir)
    signals = []
    paths = []
    for signal_path in sorted(data_dir.glob("d*")):
        if signal_path.suffix in [".xls", ".tmp"] or signal_path.is_dir():
            continue
        try:
            signal = load_signal(signal_path)
            signals.append(signal)
            paths.append(signal_path)
        except Exception as e:
            print(f"Error loading {signal_path}: {e}")
            continue
    
    if return_paths:
        return np.array(signals), paths
    return np.array(signals)
