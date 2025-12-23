import numpy as np
import matplotlib.pyplot as plt
import pywt
from pathlib import Path
from typing import List, Optional, Tuple, Union


def dwt_frequency_ranges(fs: float, levels: int) -> List[Tuple[float, float]]:
    """
    Вычисление частотных диапазонов для каждого уровня DWT.

    Parameters
    ----------
    fs : float
        Частота дискретизации (Гц)
    levels : int
        Количество уровней разложения

    Returns
    -------
    ranges : List[Tuple[float, float]]
        Список кортежей (нижняя_частота, верхняя_частота) для каждого уровня
    """
    ranges = []
    for j in range(1, levels + 1):
        f_low = fs / (2 ** (j + 1))
        f_high = fs / (2**j)
        ranges.append((f_low, f_high))
    return ranges


def dwt_scalogram(signal: np.ndarray, wavelet: str, level: Optional[int] = None) -> np.ndarray:
    """
    Построение скалограммы DWT.

    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал
    wavelet : str
        Имя базисного вейвлета
    level : int, optional
        Максимальный уровень разложения (если None, вычисляется автоматически)

    Returns
    -------
    scalogram : np.ndarray
        Массив формы (levels, len(signal)) с интерполированными коэффициентами
    """
    if level is None:
        wavelet_obj = pywt.Wavelet(wavelet)
        level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    details = coeffs[1:]  # D1 ... Dj

    scalogram = []
    for d in details:
        scalogram.append(
            np.abs(np.interp(np.linspace(0, len(d), len(signal)), np.arange(len(d)), d))
        )

    return np.array(scalogram)


def plot_dwt_scalogram(
    signal: np.ndarray,
    wavelets: Union[str, List[str]],
    sampling_rate: float = 100.0,
    hr_range: Tuple[float, float] = (0.8, 1.6),
    figsize: Tuple[int, int] = (12, 9),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Визуализация DWT скалограммы для одного или нескольких вейвлетов.

    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал
    wavelets : str or List[str]
        Имя вейвлета или список вейвлетов
    sampling_rate : float
        Частота дискретизации (Гц), по умолчанию 100.0
    hr_range : Tuple[float, float]
        Диапазон частот ЧСС для выделения (нижняя, верхняя), по умолчанию (0.8, 1.6)
    figsize : Tuple[int, int]
        Размер фигуры
    show : bool
        Показывать график сразу
    save_path : str or Path, optional
        Путь для сохранения графика

    Returns
    -------
    fig : matplotlib.figure.Figure
        Объект фигуры
    """
    if isinstance(wavelets, str):
        wavelets = [wavelets]

    n_wavelets = len(wavelets)
    plt.figure(figsize=figsize)

    for i, w in enumerate(wavelets, 1):
        try:
            S = dwt_scalogram(signal, w)
            levels = S.shape[0]

            # частотные диапазоны
            freq_ranges = dwt_frequency_ranges(sampling_rate, levels)

            plt.subplot(n_wavelets, 1, i)
            plt.imshow(S, aspect="auto", cmap="jet", origin="lower")

            # подписи уровней с частотами
            yticks = np.arange(levels)
            ylabels = [
                f"D{j + 1}: {freq_ranges[j][0]:.2f}–{freq_ranges[j][1]:.2f} Гц"
                for j in range(levels)
            ]
            plt.yticks(yticks, ylabels)

            # выделение уровня ЧСС
            for j, (f_low, f_high) in enumerate(freq_ranges):
                if hr_range[0] <= f_high and f_low <= hr_range[1]:
                    plt.axhline(j, color="white", linestyle="--", linewidth=1)

            plt.colorbar(label="|Коэффициент|")
            plt.title(f"DWT скалограмма ({w})")
            plt.xlabel("Ts")
        except Exception as e:
            print(f"Ошибка при построении скалограммы для {w}: {e}")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранен: {save_path}")

    if show:
        plt.show()
    else:
        return plt.gcf()


def process_signal_dwt(
    signal: np.ndarray,
    wavelet: str,
    level: Optional[int] = None,
    return_coeffs: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """
    Обработка сигнала DWT с возможностью возврата коэффициентов.

    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал
    wavelet : str
        Имя базисного вейвлета
    level : int, optional
        Максимальный уровень разложения
    return_coeffs : bool
        Возвращать также исходные коэффициенты разложения

    Returns
    -------
    scalogram : np.ndarray
        Скалограмма
    coeffs : List[np.ndarray], optional
        Исходные коэффициенты разложения (если return_coeffs=True)
    """
    if level is None:
        wavelet_obj = pywt.Wavelet(wavelet)
        level = pywt.dwt_max_level(len(signal), wavelet_obj.dec_len)

    coeffs = pywt.wavedec(signal, wavelet, level=level)
    details = coeffs[1:]

    scalogram = []
    for d in details:
        scalogram.append(
            np.abs(np.interp(np.linspace(0, len(d), len(signal)), np.arange(len(d)), d))
        )

    scalogram = np.array(scalogram)

    if return_coeffs:
        return scalogram, coeffs
    return scalogram


def batch_process_signals(
    signals: Union[np.ndarray, List[np.ndarray]],
    signal_names: Optional[List[str]] = None,
    wavelets: Union[str, List[str]] = "db4",
    output_dir: Optional[Union[str, Path]] = None,
    sampling_rate: float = 100.0,
    save_scalograms: bool = True,
    save_coefficients: bool = False,
    plot_scalograms: bool = False,
    level: Optional[int] = None,
) -> dict:
    """
    Пакетная обработка сигналов DWT.

    Parameters
    ----------
    signals : np.ndarray or List[np.ndarray]
        Массив сигналов формы (N, L) или список массивов
    signal_names : List[str], optional
        Имена сигналов (если None, генерируются автоматически)
    wavelets : str or List[str]
        Имя вейвлета или список вейвлетов
    output_dir : str or Path, optional
        Директория для сохранения результатов
    sampling_rate : float
        Частота дискретизации (Гц)
    save_scalograms : bool
        Сохранять скалограммы в .npy файлы
    save_coefficients : bool
        Сохранять исходные коэффициенты DWT
    plot_scalograms : bool
        Строить и сохранять графики скалограмм
    level : int, optional
        Максимальный уровень разложения

    Returns
    -------
    results : dict
        Словарь с результатами обработки
    """
    # Преобразование входных данных
    if isinstance(signals, np.ndarray):
        if signals.ndim == 1:
            signals = [signals]
        else:
            signals = [signals[i] for i in range(signals.shape[0])]

    if isinstance(wavelets, str):
        wavelets = [wavelets]

    if signal_names is None:
        signal_names = [f"signal_{i:04d}" for i in range(len(signals))]

    if len(signals) != len(signal_names):
        raise ValueError(
            f"Количество сигналов ({len(signals)}) не совпадает с количеством имен ({len(signal_names)})"
        )

    # Создание выходной директории
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "scalograms": {},
        "coefficients": {},
        "saved_files": [],
    }

    # Обработка каждого сигнала и вейвлета
    for signal, signal_name in zip(signals, signal_names):
        results["scalograms"][signal_name] = {}
        results["coefficients"][signal_name] = {}

        for wavelet in wavelets:
            try:
                # Вычисление скалограммы
                if save_coefficients:
                    scalogram, coeffs = process_signal_dwt(
                        signal, wavelet, level=level, return_coeffs=True
                    )
                    results["coefficients"][signal_name][wavelet] = coeffs
                else:
                    scalogram = process_signal_dwt(signal, wavelet, level=level)

                results["scalograms"][signal_name][wavelet] = scalogram

                # Сохранение результатов
                if output_dir is not None:
                    wavelet_dir = output_dir / wavelet
                    wavelet_dir.mkdir(parents=True, exist_ok=True)

                    # Сохранение скалограммы
                    if save_scalograms:
                        scalogram_path = wavelet_dir / f"{signal_name}_scalogram.npy"
                        np.save(scalogram_path, scalogram)
                        results["saved_files"].append(scalogram_path)

                    # Сохранение коэффициентов
                    if save_coefficients:
                        coeffs_path = wavelet_dir / f"{signal_name}_coeffs.npz"
                        arr_dict = {"cA": coeffs[0]}
                        for i, detail in enumerate(coeffs[1:], 1):
                            arr_dict[f"cD{i}"] = detail
                        np.savez_compressed(coeffs_path, **arr_dict)
                        results["saved_files"].append(coeffs_path)

                    # Построение и сохранение графиков
                    if plot_scalograms:
                        plot_path = wavelet_dir / f"{signal_name}_scalogram.png"
                        plot_dwt_scalogram(
                            signal,
                            wavelet,
                            sampling_rate=sampling_rate,
                            show=False,
                            save_path=plot_path,
                        )
                        plt.close()
                        results["saved_files"].append(plot_path)

            except Exception as e:
                print(f"Ошибка при обработке {signal_name} с вейвлетом {wavelet}: {e}")
                continue

    return results


def load_scalogram(file_path: Union[str, Path]) -> np.ndarray:
    """
    Загрузка сохраненной скалограммы.

    Parameters
    ----------
    file_path : str or Path
        Путь к файлу .npy

    Returns
    -------
    scalogram : np.ndarray
        Загруженная скалограмма
    """
    return np.load(file_path)


def plot_dwt_scalogram_plotly(
    signal: np.ndarray,
    wavelet: str,
    sampling_rate: float = 100.0,
    hr_range: Tuple[float, float] = (0.8, 1.6),
    title: Optional[str] = None,
) -> dict:
    """
    Создание данных для интерактивной визуализации скалограммы DWT с помощью Plotly.

    Parameters
    ----------
    signal : np.ndarray
        Входной сигнал
    wavelet : str
        Имя базисного вейвлета
    sampling_rate : float
        Частота дискретизации (Гц), по умолчанию 100.0
    hr_range : Tuple[float, float]
        Диапазон частот ЧСС для выделения (нижняя, верхняя), по умолчанию (0.8, 1.6)
    title : str, optional
        Заголовок графика

    Returns
    -------
    fig_data : dict
        Словарь с данными для построения Plotly графика:
        - 'scalogram': np.ndarray - данные скалограммы
        - 'freq_ranges': List[Tuple[float, float]] - частотные диапазоны
        - 'hr_levels': List[int] - индексы уровней в диапазоне ЧСС
        - 'ylabels': List[str] - подписи для осей Y
    """
    S = dwt_scalogram(signal, wavelet)
    levels = S.shape[0]

    # Частотные диапазоны
    freq_ranges = dwt_frequency_ranges(sampling_rate, levels)

    # Определяем уровни в диапазоне ЧСС
    hr_levels = []
    for j, (f_low, f_high) in enumerate(freq_ranges):
        if hr_range[0] <= f_high and f_low <= hr_range[1]:
            hr_levels.append(j)

    # Подписи для осей
    ylabels = [
        f"D{j + 1}: {freq_ranges[j][0]:.2f}–{freq_ranges[j][1]:.2f} Гц" for j in range(levels)
    ]

    return {
        "scalogram": S,
        "freq_ranges": freq_ranges,
        "hr_levels": hr_levels,
        "ylabels": ylabels,
        "wavelet": wavelet,
        "title": title or f"DWT скалограмма ({wavelet})",
    }
