import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.paths import LOGS_PATH, DATA_VIS_PATH, DATA_PATH


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "extension_name",
        type=str,
        help="Experiment extension name"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    extension_name = args.extension_name

    thresholds = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
    f1_mean, f1_std = [], []
    precision_mean, precision_std = [], []
    recall_mean, recall_std = [], []

    csv_pattern = os.path.join(LOGS_PATH, f"dl_evaluation_{extension_name}", f"dl_evaluation_{extension_name}_{{}}.csv")
    for threshold in thresholds:
        performance_csv_filename = csv_pattern.format(threshold)
        perf_df = pd.read_csv(performance_csv_filename)

        mean_row = perf_df.iloc[-2]
        std_row = perf_df.iloc[-1]

        # Precision, Recall, F1Score
        f1_mean.append(mean_row['F1Score'])
        f1_std.append(std_row['F1Score'])
        precision_mean.append(mean_row['Precision'])
        precision_std.append(std_row['Precision'])
        recall_mean.append(mean_row['Recall'])
        recall_std.append(std_row['Recall'])

    f1_mean, f1_std = np.array(f1_mean), np.array(f1_std)
    precision_mean, precision_std = np.array(precision_mean), np.array(precision_std)
    recall_mean, recall_std = np.array(recall_mean), np.array(recall_std)

    df_to_save = pd.DataFrame({'threshold': thresholds,
                               'f1score_mean': f1_mean, 'f1score_std': f1_std,
                               'precision_mean': precision_mean, 'precision_std': precision_std,
                               'recall_mean': recall_mean, 'recall_std': recall_std})
    csv_filename = os.path.join(DATA_PATH, f"performance_threshold_{extension_name}.csv")
    df_to_save.to_csv(csv_filename, index=False)

    plt.figure()
    plt.plot(thresholds, f1_mean, color='k', linestyle='-', label='F1Score')
    plt.plot(thresholds, precision_mean, color='k', linestyle='--', label='Precision')
    plt.plot(thresholds, recall_mean, color='k', linestyle=':', label='Recall')

    f1_high, f1_low = f1_mean+f1_std, f1_mean-f1_std
    plt.fill_between(thresholds, f1_high, f1_low, where=f1_high >= f1_low, facecolor='#fccfcf', interpolate=True, alpha=0.5)

    precision_high, precision_low = precision_mean+precision_std, precision_mean-precision_std
    plt.fill_between(thresholds, precision_high, precision_low, where=precision_high >= precision_low, facecolor='#cfeffc', interpolate=True, alpha=0.5)

    recall_high, recall_low = recall_mean+recall_std, recall_mean-recall_std
    plt.fill_between(thresholds, recall_high, recall_low, where=recall_high >= recall_low, facecolor='#d6ffd1', interpolate=True, alpha=0.5)

    plt.xlabel('Threshold')
    plt.xticks(thresholds[1::2])
    plt.yticks(np.arange(0.1, 1, 0.1))
    plt.ylim(0, 1)

    plt.grid(alpha=0.3)

    plt.legend()
    plot_filename = os.path.join(DATA_VIS_PATH, f"performance_threshold_{extension_name}.png")
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0.0)
