import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    categories = ["background", "building", "green space", "blue space",
                  "road", "vehicle", "field", "on water"]
    Prithvi_results_5epochs = [0.5407, np.nan, 0.0000, 0.0000,  np.nan, 0.0055, np.nan, 0.0010]
    UNetFormer_results_5epochs = [0.7873, 0.0166, 0.1136, 0.0369, 0.0087, 0.0953, 0.0227, 0.0834]

    nan_indices = [i for i, value in enumerate(Prithvi_results_5epochs) if np.isnan(value)]
    small_value_indices = [i for i, value in enumerate(Prithvi_results_5epochs) if value == 0.0]
    Prithvi_results_5epochs = pd.Series(Prithvi_results_5epochs).fillna(0).tolist()

    x = np.arange(len(categories))
    width = 0.35

    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(
        x - width / 2, Prithvi_results_5epochs, width, label='Prithvi', color='blue', edgecolor='black'
    )
    bars2 = plt.bar(
        x + width / 2, UNetFormer_results_5epochs, width, label='UNetFormer', color='orange', edgecolor='black'
    )

    for i in nan_indices:
        plt.text(
            x[i] - width / 2, 0.005,
            "NaN", ha='center', va='bottom', weight='bold', fontsize=7, color='red',  rotation=75
        )

    for i in small_value_indices:
        plt.text(
            x[i] - width / 2, 0.005,
            "Negligible IoU detected",
            ha='center', va='bottom', weight='bold', fontsize=7, color='blue', rotation=75
        )

    plt.xlabel('Categories')
    plt.ylabel('IoU Results')
    plt.title('Prithvi vs UNetFormer for DOTA')
    plt.xticks(x, categories, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()