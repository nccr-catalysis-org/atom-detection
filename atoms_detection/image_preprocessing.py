import numpy as np

from scipy.ndimage.filters import gaussian_filter, median_filter
from atoms_detection.fast_filters import median_filter_parallel
from PIL import Image


def preprocess_jpg(np_img: np.ndarray) -> np.ndarray:
    return np_img[:, :, 0]


def dl_prepro_image(np_img: np.ndarray, ruler_units=None, clip=1):
    # np_bg = gaussian_filter(np_img, sigma=20)
    if len(np_img.shape) == 3:
        np_img = preprocess_jpg(np_img)
    scale_factor = None
    if ruler_units is not None:
        try:
            ruler_size = get_ruler_size(np_img)
            np_img, scale_factor = rescale_img_to_target_pxls_nm(
                np_img, ruler_size, ruler_units
            )
        except Exception:
            pass

    print("WARNING, MANUAL CLIP USAGE")
    clip = 0.999
    max_allowed = np.quantile(np_img, q=clip)
    np_img = np.clip(np_img, a_min=0, a_max=max_allowed)
    try:
        np_bg = median_filter_parallel(np_img, 40, splits=4)
    except Exception as e:
        print(e)
        print("Median filter failed, using slower scipy version")
        np_bg = median_filter(np_img, 40)
    np_clean = np_img - np_bg
    np_clean[np_clean < 0] = 0
    np_normed = (np_clean - np_clean.min()) / (np_clean.max() - np_clean.min())
    # np_normed = (np_img - np_img.min()) / (np_img.max() - np_img.min())
    from matplotlib import pyplot as plt

    if scale_factor is not None:
        return np_normed, scale_factor
    return np_normed


def cv_prepro_image(img: np.ndarray) -> np.ndarray:
    bg_img = gaussian_filter(img, sigma=10)
    clean_img = img - bg_img
    normed_img = (clean_img - clean_img.min()) / (clean_img.max() - clean_img.min())
    return normed_img


def get_ruler_size(img: np.ndarray) -> int:
    ruler_start_location_percent = 0.0625  # empirically located here in samples
    ruler_start_coords = int(
        img.shape[0] * (1 - ruler_start_location_percent) - 1
    ), int(img.shape[1] * ruler_start_location_percent)
    if img[ruler_start_coords] != img.max():
        print("Ruler start position verification failed, skipping rescaling")
        raise Exception
    else:
        ruler_iterator = ruler_start_coords
        while img[ruler_iterator] == img[ruler_start_coords]:
            ruler_iterator = ruler_iterator[0], ruler_iterator[1] + 1
        return ruler_iterator[1] - ruler_start_coords[1]


def rescale_img_to_target_pxls_nm(
    img: np.ndarray, ruler_pixels: int, ruler_units: int, atom_prior=None
):
    target_scale = (
        512 / 15
    )  # original images were 512x512 and labelled 15nm across, 34 pixels per nano
    pixels_per_nanometer = ruler_pixels / ruler_units  # current pixels per nano
    scaling_factor = target_scale / pixels_per_nanometer
    new_dimensions = int(img.shape[0] * scaling_factor), int(
        img.shape[1] * scaling_factor
    )
    if atom_prior is None:
        return np.array(Image.fromarray(img).resize(new_dimensions)), scaling_factor
    else:
        raise NotImplementedError
