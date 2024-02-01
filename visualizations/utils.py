
from PIL import Image
import numpy as np
from networkx.drawing.tests.test_pylab import plt

from atoms_detection.image_preprocessing import preprocess_jpg
from utils.constants import Split


def plot_gt_pred_on_img(img_normed, gt_coords, pred_coords):
    imgsize = img_normed.shape[0]/512
    plt.figure(figsize=(imgsize*10, imgsize*10))
    plt.axis('off')
    plt.imshow(img_normed)
    if pred_coords is not None:
        x, y = zip(*pred_coords)
        plt.scatter(x, y, s=300, linewidths=3, c='#FFDB1A', marker='+')
    if gt_coords is not None:
        gt_x, gt_y = zip(*gt_coords)
        plt.scatter(gt_x, gt_y, s=300, linewidths=2, facecolors='none', edgecolors='r')
