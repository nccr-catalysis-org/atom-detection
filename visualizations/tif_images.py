import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from utils.paths import HAADF_DATASET, IMG_PATH, ORIG_VIS_PATH


if __name__=="__main__":
    df = pd.read_csv(HAADF_DATASET)
    for image_name in df['Filename']:
        image_filename = os.path.join(IMG_PATH, image_name)
        img = Image.open(image_filename)
        img_arr = np.array(img).astype(np.float32)
        img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
        plt.figure()
        plt.axis('off')
        plt.imshow(img_normed)
        vis_path = os.path.join(ORIG_VIS_PATH, '{}.png'.format(os.path.splitext(image_name)[0]))
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()
