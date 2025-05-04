import numpy as np
import pandas as pd
from scipy.stats import f
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Реальные параметры
sample_count = 10000
real_dispersion1 = 2
real_dispersion2 = 1
real_mu1 = 0
real_mu2 = 0
real_tau = real_dispersion1 / real_dispersion2
alpha = 0.05
real_standard_deviation1 = np.sqrt(real_dispersion1)
real_standard_deviation2 = np.sqrt(real_dispersion2)

def generate_confidence_interval(sample_count, size):
    lows = []
    highs = []
    coverage = 0  # Счетчик покрытий реального значения
    
    for _ in range(sample_count):
        # Выборка данных из нормального распределения
        X1 = np.random.normal(real_mu1, real_standard_deviation1, size)
        X2 = np.random.normal(real_mu2, real_standard_deviation2, size)

        # Выборка дисперсий (используем size вместо size-1 как в исходном коде)
        S1 = np.sum((X1 - real_mu1)**2) / size
        S2 = np.sum((X2 - real_mu2)**2) / size

        # Распределение Фишера (оставляем size как в исходном коде)
        F_low = f.ppf(alpha / 2, size, size)
        F_high = f.ppf(1 - alpha / 2, size, size)

        # Границы доверительного интервала
        Confintlow = (S1 / S2) / F_high
        Cofinthigh = (S1 / S2) / F_low

        lows.append(Confintlow)
        highs.append(Cofinthigh)
        
        # Проверяем покрытие реального параметра
        if Confintlow <= real_tau <= Cofinthigh:
            coverage += 1
    
    # Создаем DataFrame с результатами
    Confint_df = pd.DataFrame({
        'Lower': lows,
        'Upper': highs
    })
    
    # Вычисляем долю покрытий
    coverage_prob = coverage / sample_count
    return coverage_prob, Confint_df

# Проводим эксперименты для двух размеров выборок
coverage_25, df_25 = generate_confidence_interval(sample_count, 25)
coverage_10000, df_10000 = generate_confidence_interval(sample_count, 10000)

# Функция для рисования boxplot
def draw_plots(df_small, df_large, real_tau):
    # Преобразуем данные в long-form
    df_small_long = df_small.melt(var_name='boundary', value_name='value')
    df_large_long = df_large.melt(var_name='boundary', value_name='value')
    
    # Определяем цвета с правильным регистром
    palette = {'Lower': 'blue', 'Upper': 'green'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7))
    fig.suptitle(f"Boxplot границ доверительного интервала уровня доверия {1 - alpha:.0%}", 
                 fontsize=14, y=1.03)
    
    # Boxplot для малой выборки
    sns.boxplot(data=df_small_long, x='boundary', y='value', ax=ax1, palette=palette)
    ax1.axhline(y=real_tau, color='red', linestyle='--', label=f'Истинное tau: {real_tau}')
    ax1.set_title(f"Размер выборки: 25 (покрытий: {coverage_25:.2%})")
    ax1.set_ylabel("Значение границ интервала")
    ax1.set_xlabel("")  # <-- Убираем подпись снизу
    ax1.legend()
    
    # Boxplot для большой выборки
    sns.boxplot(data=df_large_long, x='boundary', y='value', ax=ax2, palette=palette)
    ax2.axhline(y=real_tau, color='red', linestyle='--', label=f'Истинное tau: {real_tau}')
    ax2.set_title(f"Размер выборки: 10000 (покрытий: {coverage_10000:.2%})")
    ax2.set_ylabel("")
    ax2.set_xlabel("")  # <-- Убираем подпись снизу
    ax2.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Вызываем функцию для рисования графиков
draw_plots(df_25, df_10000, real_tau)
