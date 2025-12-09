import numpy as np
from scipy.optimize import minimize
from scipy import sparse


def ssa_decompose(signal, window_size=None, n_components=5, fast=False):
    """
    Разложение сигнала методом SSA (Singular Spectrum Analysis).

    Parameters:
    -----------
    signal : array-like
        Входной сигнал
    window_size : int, optional
        Размер окна для траекторной матрицы. Если None, используется len(signal) // 2
    n_components : int
        Количество компонент для восстановления

    Returns:
    --------
    components : list of arrays
        Список восстановленных компонент, упорядоченных по убыванию важности
        (по сингулярным значениям). Первая компонента (components[0]) - самая важная.
    """
    signal = np.array(signal, dtype=float)
    N = len(signal)

    # Проверка минимальной длины сигнала
    if N < 4:
        raise ValueError(f"Сигнал слишком короткий для SSA. Минимальная длина: 4, получено: {N}")

    # Определяем размер окна
    if window_size is None:
        window_size = max(N // 2, 2)  # Минимум 2 для создания матрицы
    window_size = min(window_size, N // 2)
    window_size = max(window_size, 2)  # Минимум 2

    # Шаг 1: Создание траекторной матрицы (embedding)
    K = N - window_size + 1
    X = np.zeros((window_size, K))
    for i in range(K):
        X[:, i] = signal[i : i + window_size]

    # Шаг 2: SVD разложение
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Шаг 3: Группировка и восстановление компонент
    components = []
    for i in range(min(n_components, len(s))):
        # Создаем матрицу для i-й компоненты
        Xi = s[i] * np.outer(U[:, i], Vt[i, :])

        # Восстановление сигнала из матрицы (диагональное усреднение)
        if fast:
            component = _diagonal_averaging_fast(Xi, L=window_size, K=K)
        else:
            component = _diagonal_averaging(Xi)
        components.append(component)

    return components


def get_ssa_singular_values(signal, window_size=None):
    """
    Получить сингулярные значения SSA разложения.
    Полезно для анализа важности компонент.

    Parameters:
    -----------
    signal : array-like
        Входной сигнал
    window_size : int, optional
        Размер окна для траекторной матрицы. Если None, используется len(signal) // 2

    Returns:
    --------
    singular_values : array
        Сингулярные значения, упорядоченные по убыванию
    """
    signal = np.array(signal, dtype=float)
    N = len(signal)

    # Проверка минимальной длины сигнала
    if N < 4:
        raise ValueError(f"Сигнал слишком короткий для SSA. Минимальная длина: 4, получено: {N}")

    # Определяем размер окна
    if window_size is None:
        window_size = max(N // 2, 2)
    window_size = min(window_size, N // 2)
    window_size = max(window_size, 2)

    # Создание траекторной матрицы
    K = N - window_size + 1
    X = np.zeros((window_size, K))
    for i in range(K):
        X[:, i] = signal[i : i + window_size]

    # SVD разложение
    _, s, _ = np.linalg.svd(X, full_matrices=False)

    return s


def _diagonal_averaging(X):
    """
    Восстановление сигнала из траекторной матрицы методом диагонального усреднения.

    Parameters:
    -----------
    X : array-like, shape (L, K)
        Траекторная матрица

    Returns:
    --------
    signal : array
        Восстановленный сигнал
    """
    L, K = X.shape
    N = L + K - 1
    signal = np.zeros(N)
    counts = np.zeros(N)

    # Проходим по всем элементам матрицы и суммируем по диагоналям
    for i in range(L):
        for j in range(K):
            n = i + j
            if n < N:
                signal[n] += X[i, j]
                counts[n] += 1

    # Усредняем
    signal = signal / np.maximum(counts, 1)

    return signal


def _diagonal_averaging_fast(X, L, K):
    """
    Быстрое восстановление сигнала из траекторной матрицы методом диагонального усреднения.

    Parameters:
    -----------
    X : array-like, shape (L, K)
        Траекторная матрица
    L : int
        Количество строк (window_size)
    K : int
        Количество столбцов (N - window_size + 1)

    Returns:
    --------
    signal : array
        Восстановленный сигнал длины N = L + K - 1
    """
    N = L + K - 1

    # Создаем разреженную матрицу для суммирования
    rows = np.repeat(np.arange(L), K)
    cols = np.tile(np.arange(K), L)
    data = X.ravel()

    # Суммирующая матрица (N x (L*K))
    # Каждый столбец соответствует одному элементу X
    # Каждая строка соответствует одной диагонали
    diag_indices = rows + cols

    # Создаем разреженную матрицу для суммирования
    summing_matrix = sparse.csr_matrix(
        (np.ones_like(data), (diag_indices, np.arange(len(data)))), shape=(N, len(data))
    )

    # Суммируем
    signal = summing_matrix.dot(data)
    counts = summing_matrix.dot(np.ones_like(data))

    # Усредняем
    signal = np.where(counts > 0, signal / counts, 0)

    return signal


def _sinusoid_model(t, a, w, h):
    return a * np.sin(w * t + h)


def _loss_function(params, t, y_true):
    a, w, h = params
    y_pred = _sinusoid_model(t, a, w, h)
    return np.sum((y_true - y_pred) ** 2)


def fit_sinusoid(signal, sampling_rate=100.0, initial_freq=None):
    """
    Аппроксимация сигнала синусоидой y(t) = a*sin(w*t+h) методом МНК.

    Parameters:
    -----------
    signal : array-like
        Входной сигнал (главная гармоника)
    sampling_rate : float
        Частота дискретизации (Гц)
    initial_freq : float, optional
        Начальное приближение частоты (Гц). Если None, используется FFT

    Returns:
    --------
    frequency : float
        Частота в Гц
    params : tuple
        Параметры (a, w, h) - амплитуда, угловая частота, фаза
    """
    signal = np.array(signal, dtype=float)
    N = len(signal)

    # Проверка на пустой сигнал
    if N == 0:
        raise ValueError("Сигнал не может быть пустым")
    if N < 3:
        raise ValueError(
            f"Сигнал слишком короткий для аппроксимации. Минимальная длина: 3, получено: {N}"
        )

    t = np.arange(N) / sampling_rate  # Временные точки в секундах

    # Определяем начальные параметры
    # Амплитуда - используем RMS значение
    signal_std = np.std(signal)
    a_init = signal_std * np.sqrt(2) if signal_std > 0 else 1.0

    # Угловая частота
    if initial_freq is None:
        # Используем FFT для начального приближения
        fft_values = np.fft.fft(signal)
        fft_magnitude = np.abs(fft_values)
        frequencies = np.fft.fftfreq(N, 1 / sampling_rate)

        positive_freq_idx = (frequencies > 0) & (frequencies < 3.0)  # Ограничиваем до 3 Гц
        if np.any(positive_freq_idx):
            positive_frequencies = frequencies[positive_freq_idx]
            positive_magnitude = fft_magnitude[positive_freq_idx]
            main_freq_idx = np.argmax(positive_magnitude)
            initial_freq = positive_frequencies[main_freq_idx]
        else:
            initial_freq = 1.0  # Значение по умолчанию (60 BPM)

    # Ограничиваем начальную частоту разумными пределами
    initial_freq = np.clip(initial_freq, 0.5, 3.0)
    w_init = 2 * np.pi * initial_freq  # Угловая частота

    # Фаза - начинаем с нуля
    h_init = 0.0

    # Границы для оптимизации
    # Амплитуда: от 0 до 5*std (с запасом)
    signal_max_amplitude = max(np.abs(signal).max(), signal_std * 3) if signal_std > 0 else 1.0
    # Угловая частота: от 0.67 до 3 Гц (40-180 BPM) - типичный диапазон ЧСС
    # Фаза: от -2π до 2π
    bounds = [
        (0, signal_max_amplitude * 2),  # амплитуда
        (2 * np.pi * 0.67, 2 * np.pi * 3.0),  # угловая частота (0.67-3 Гц = 40-180 BPM)
        (-2 * np.pi, 2 * np.pi),  # фаза
    ]

    # Оптимизация методом наименьших квадратов
    result = minimize(
        _loss_function,
        x0=[a_init, w_init, h_init],
        args=(t, signal),
        method="L-BFGS-B",
        bounds=bounds,
    )

    a_opt, w_opt, h_opt = result.x

    # Преобразуем угловую частоту в обычную частоту (Гц)
    frequency = w_opt / (2 * np.pi)

    return frequency, (a_opt, w_opt, h_opt)


def compute_ssa_frequencies(signals, sampling_rate=100.0, n_components=5):
    """
    Вычисление частот главной гармоники сигналов методом SSA.

    Parameters:
    -----------
    signals : array-like
        Массив сигналов или один сигнал
    sampling_rate : float
        Частота дискретизации (Гц)
    n_components : int
        Количество компонент SSA для использования

    Returns:
    --------
    frequencies : array
        Массив частот в Гц для каждого сигнала
    """
    signals = np.array(signals)

    # Обработка одного сигнала
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    frequencies = []

    for signal in signals:
        # Шаг 1: SSA разложение
        components = ssa_decompose(signal, n_components=n_components)

        # Шаг 2: Берем первую (главную) компоненту
        main_component = components[0]

        # Шаг 3: Аппроксимируем главную компоненту синусоидой
        frequency, _ = fit_sinusoid(main_component, sampling_rate=sampling_rate)

        frequencies.append(frequency)

    return np.array(frequencies)
