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

#    thresholds = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

    thresholds = np.array([0.89])
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

    print(f1_mean, precision_mean, recall_mean)

#    df_to_save = pd.DataFrame({'threshold': thresholds,
#                               'f1score_mean': f1_mean, 'f1score_std': f1_std,
#                               'precision_mean': precision_mean, 'precision_std': precision_std,
#                               'recall_mean': recall_mean, 'recall_std': recall_std})
#    csv_filename = os.path.join(DATA_PATH, f"performance_threshold_{extension_name}.csv")
#    df_to_save.to_csv(csv_filename, index=False)
