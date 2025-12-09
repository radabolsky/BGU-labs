import matplotlib.pyplot as plt
import numpy as np


def plot_frequency_comparison(fourier_freqs, ssa_freqs, save_path=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(fourier_freqs, ssa_freqs, alpha=0.6, s=50)

    min_freq = min(np.min(fourier_freqs), np.min(ssa_freqs))
    max_freq = max(np.max(fourier_freqs), np.max(ssa_freqs))

    plt.plot(
        [min_freq, max_freq], [min_freq, max_freq], "r--", linewidth=2, label="Биссектриса (y=x)"
    )

    plt.xlabel("Частота по Фурье (Гц)", fontsize=12)
    plt.ylabel("Частота по SSA (Гц)", fontsize=12)
    plt.title("Сравнение частот главных компонент: Фурье vs SSA", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
