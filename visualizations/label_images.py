import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from atoms_detection.dataset import CoordinatesDataset
from utils.paths import PT_DATASET, IMG_PATH, COORDS_PATH, LABEL_VIS_PATH

df = pd.read_csv(PT_DATASET)
for idx, row in df.iterrows():
# row = list(df.iterrows())[0][1]
    image_name = row['Image']
    coords_name = row['Coords']
    image_filename = os.path.join(IMG_PATH, image_name)
    coords_filename = os.path.join(COORDS_PATH, coords_name)
    img = Image.open(image_filename)
    atom_coordinates = pd.read_csv(coords_filename)
    x, y = atom_coordinates['X'], atom_coordinates['Y']
    # coords = CoordinatesDataset.load_coordinates(coords_filename)

    img_arr = np.array(img).astype(np.float32)
    img_normed = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min())
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(img_normed)
    plt.scatter(x, y, s=80, linewidths=1.5, c='#FFDB1A', marker='+')
    vis_path = os.path.join(LABEL_VIS_PATH, '{}.png'.format(os.path.splitext(image_name)[0]))
    # plt.show()
    plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0, transparent=True)
    plt.close()
