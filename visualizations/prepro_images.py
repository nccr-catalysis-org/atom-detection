import os

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from atoms_detection.image_preprocessing import dl_prepro_image
from utils.paths import PT_DATASET, FE_DATASET, IMG_PATH, PREPRO_VIS_PATH


def generate_prepro_plots(dataset_path: str, vis_folder: str):
    if not os.path.exists(vis_folder):
        os.makedirs(vis_folder)

    df = pd.read_csv(dataset_path)
    for image_name in df['Filename']:
        image_filename = os.path.join(IMG_PATH, image_name)
        img = Image.open(image_filename)
        np_img = np.array(img).astype(np.float32)
        np_prepro_img = dl_prepro_image(np_img)
        plt.figure()
        plt.axis('off')
        plt.imshow(np_prepro_img)
        vis_path = os.path.join(vis_folder, '{}.png'.format(os.path.splitext(image_name)[0]))
        plt.savefig(vis_path, bbox_inches='tight', pad_inches=0.0)
        plt.close()


if __name__ == "__main__":
    generate_prepro_plots(PT_DATASET, os.path.join(PREPRO_VIS_PATH, 'Pt-Catalyst'))
    generate_prepro_plots(FE_DATASET, os.path.join(PREPRO_VIS_PATH, 'Fe-Catalyst'))
