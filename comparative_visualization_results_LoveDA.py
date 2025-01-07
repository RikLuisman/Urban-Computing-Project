import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    categories = ["background", "building", "road", "water",
                  "forest", "agriculture", "barren"]
    Prithvi_results_urban = [0.4341, 0.0995, 0.0000, 0.1861, 0.0230,  0.0000, 0.0461]
    Prithvi_results_rural = [0.2474, 0.0319, 0.0000, 0.0493, 0.3671, 0.3975, 0.0404]
    UNetFormer_results = [0.447, 0.588, 0.549, 0.796, 0.460, 0.625, 0.201]

    small_value_indices_urban = [i for i, value in enumerate(Prithvi_results_urban) if value == 0.0]
    small_value_indices_rural = [i for i, value in enumerate(Prithvi_results_rural) if value == 0.0]

    x = np.arange(len(categories))
    width = 0.25

    plt.figure(figsize=(8, 6))
    bars1 = plt.bar(
        x - width, Prithvi_results_urban, width, label='Prithvi - LoveDA Urban', color='dodgerblue', edgecolor='black'
    )
    bars2 = plt.bar(
        x, Prithvi_results_rural, width, label='Prithvi - LoveDA Rural', color='teal', edgecolor='black'
    )
    bars3 = plt.bar(
        x + width, UNetFormer_results, width, label='UNetFormer - LoveDA', color='orange', edgecolor='black'
    )


    for i in small_value_indices_urban:
        plt.text(
            x[i] - width, 0.005,
            "Negligible IoU detected",
            ha='center', va='bottom', fontsize=7, weight='bold', color='dodgerblue', rotation=80
        )

    for i in small_value_indices_rural:
        plt.text(
            x[i], 0.005,
            "Negligible IoU detected",
            ha='center', va='bottom', fontsize=7, weight='bold', color='teal', rotation=80
        )

    plt.xlabel('Categories')
    plt.ylabel('IoU Results')
    plt.title('Prithvi vs UNetFormer for LoveDA')
    plt.xticks(x, categories, rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()