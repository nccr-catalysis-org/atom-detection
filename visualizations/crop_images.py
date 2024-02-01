import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from utils.paths import CROPS_VIS_PATH

df = pd.read_csv("dataset/atom_crops_replicate.csv")
for crop_name in df['Filename']:
    crop_filename = os.path.join("data/atom_crops_data_sac_cnn", crop_name)
    crop = Image.open(crop_filename)
    crop_arr = np.array(crop).astype(np.float32)
    plt.figure()
    plt.axis('off')
    plt.imshow(crop_arr)
    vis_path = os.path.join(CROPS_VIS_PATH, '{}.png'.format(os.path.splitext(crop_name)[0]))
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0)
    plt.close()

