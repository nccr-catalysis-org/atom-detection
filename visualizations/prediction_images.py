import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from atoms_detection.dataset import CoordinatesDataset
from utils.constants import Split
from utils.paths import DETECTION_LOGS, IMG_PATH, PRED_VIS_PATH, PT_DATASET


if __name__ == "__main__":
    for name_file in os.listdir(DETECTION_LOGS):
        print(name_file)
        filepath = os.path.join(DETECTION_LOGS, name_file)

        image_name = os.path.splitext(name_file)[0] + ".tif"
        image_filename = os.path.join(IMG_PATH, image_name)
        img = Image.open(image_filename)

        df = pd.read_csv(filepath)
        x, y = [], []
        for idx, row in df.iterrows():
            x.append(row['x'])
            y.append(row['y'])

        img_arr = np.array(img).astype(np.float32)
        img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(img_normed)
        plt.scatter(x, y, s=300, linewidths=3, c='#FFDB1A', marker='+')

        vis_path = os.path.join(PRED_VIS_PATH, '{}.png'.format(os.path.splitext(image_name)[0]))
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
        plt.close()
