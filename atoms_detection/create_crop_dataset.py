import os

import numpy as np
import pandas as pd
from PIL import Image

from atoms_detection.image_preprocessing import dl_prepro_image
from atoms_detection.dataset import CoordinatesDataset
from utils.paths import CROPS_PATH, CROPS_DATASET, PT_DATASET
from utils.constants import Split, CropsColumns
import matplotlib.pyplot as plt # I don't know why tf but it doesn't work if not here

np.random.seed(777)

window_size = (21, 21)
halfx_window = ((window_size[0] - 1) // 2)
halfy_window = ((window_size[1] - 1) // 2)


def get_gaussian_kernel(size=21, mean=0, sigma=0.2):
    # Initializing value of x-axis and y-axis
    # in the range -1 to 1
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    dst = np.sqrt(x * x + y * y)

    # Calculating Gaussian array
    kernel = np.exp(-((dst - mean) ** 2 / (2.0 * sigma ** 2)))
    return kernel


def generate_support_img(coordinates, window_size):
    support_img = np.zeros((512, 512))
    kernel = get_gaussian_kernel(size=window_size[0])
    halfx_window = ((window_size[0] - 1) // 2)
    halfy_window = ((window_size[1] - 1) // 2)
    for x, y in coordinates:
        x_range = (x - halfx_window, x + halfx_window + 1)
        y_range = (y - halfy_window, y + halfy_window + 1)

        x_diff = [0, 0]
        y_diff = [0, 0]
        if x_range[0] < 0:
            x_diff[0] = 0 - x_range[0]
        if x_range[1] > 512:
            x_diff[1] = x_range[1] - 512
        if y_range[0] < 0:
            y_diff[0] = 0 - y_range[0]
        if y_range[1] > 512:
            y_diff[1] = y_range[1] - 512

        real_kernel = kernel[x_diff[0]:window_size[0] - x_diff[1], y_diff[0]:window_size[1] - y_diff[1]]
        real_x_crop = (x_range[0] + x_diff[0], x_range[1] - x_diff[1])
        real_y_crop = (y_range[0] + y_diff[0], y_range[1] - y_diff[1])

        support_img[real_x_crop[0]:real_x_crop[1], real_y_crop[0]:real_y_crop[1]] += real_kernel

    support_img = support_img.T
    return support_img


def open_image(img_filename):
    img = Image.open(img_filename)
    np_img = np.asarray(img).astype(np.float32)
    np_img = dl_prepro_image(np_img)
    img = Image.fromarray(np_img)
    return img


def create_crop(img: Image, x_center: int, y_center: int):
    crop_coords = (
        x_center - halfx_window,
        y_center - halfy_window,
        x_center + halfx_window + 1,
        y_center + halfy_window + 1
    )
    crop = img.crop(crop_coords)
    return crop


def create_crops_dataset(crops_folder: str, coords_csv: str, crops_dataset: str):
    if not os.path.exists(crops_folder):
        os.makedirs(crops_folder)

    crop_name_list = []
    orig_name_list = []
    x_list = []
    y_list = []
    label_list = []

    n_positives = 0
    label = 1
    dataset = CoordinatesDataset(coords_csv)
    print('Creating positive crops...')
    for data_filename, label_filename in dataset.iterate_data(Split.TRAIN):
        if label_filename is None:
            continue

        print(data_filename)
        orig_img_name = os.path.basename(data_filename)
        img_name = os.path.splitext(orig_img_name)[0]

        img = open_image(data_filename)
        coordinates = dataset.load_coordinates(label_filename)

        for x_center, y_center in coordinates:
            crop = create_crop(img, x_center, y_center)
            crop_name = "{}_{}_{}.tif".format(img_name, x_center, y_center)
            crop.save(os.path.join(crops_folder, crop_name))

            crop_name_list.append(crop_name)
            orig_name_list.append(orig_img_name)
            x_list.append(x_center)
            y_list.append(y_center)
            label_list.append(label)

            n_positives += 1

    label = 0
    no_train_images = dataset.split_length(Split.TRAIN)
    neg_crops_per_image = [n_positives // no_train_images + (1 if x < n_positives % no_train_images else 0) for x in range(no_train_images)]
    print('Creating negative crops...')
    for (data_filename, label_filename), no_neg_crops in zip(dataset.iterate_data(Split.TRAIN), neg_crops_per_image):
        print(data_filename)
        orig_img_name = os.path.basename(data_filename)
        img_name = os.path.splitext(orig_img_name)[0]
        img = open_image(data_filename)

        if label_filename:
            coordinates = dataset.load_coordinates(label_filename)
            support_map = generate_support_img(coordinates, window_size)
        else:
            support_map = None

        for _ in range(no_neg_crops):
            x_rand = np.random.randint(0, 512)
            y_rand = np.random.randint(0, 512)

            if support_map is not None:
                while support_map[x_rand, y_rand] != 0:
                    x_rand = np.random.randint(0, 512)
                    y_rand = np.random.randint(0, 512)

            x_center, y_center = x_rand, y_rand

            crop = create_crop(img, x_center, y_center)
            crop_name = "{}_{}_{}.tif".format(img_name, x_center, y_center)
            crop.save(os.path.join(crops_folder, crop_name))

            crop_name_list.append(crop_name)
            orig_name_list.append(orig_img_name)
            x_list.append(x_center)
            y_list.append(y_center)
            label_list.append(label)

    df_data = {
        CropsColumns.FILENAME: crop_name_list,
        CropsColumns.ORIGINAL: orig_name_list,
        CropsColumns.X: x_list,
        CropsColumns.Y: y_list,
        CropsColumns.LABEL: label_list
    }
    df = pd.DataFrame(df_data, columns=[
        CropsColumns.FILENAME,
        CropsColumns.ORIGINAL,
        CropsColumns.X,
        CropsColumns.Y,
        CropsColumns.LABEL
    ])

    df_pos = df[df.Label == 1]
    df_neg = df[df.Label == 0]

    pos_len = len(df_pos)
    neg_len = len(df_neg)

    pos_train, pos_val, pos_test = np.split(df_pos.sample(frac=1), [int(0.8*pos_len), int(0.9*pos_len)])
    neg_train, neg_val, neg_test = np.split(df_neg.sample(frac=1), [int(0.8*neg_len), int(0.9*neg_len)])
    pos_train[CropsColumns.SPLIT] = Split.TRAIN
    pos_val[CropsColumns.SPLIT] = Split.VAL
    pos_test[CropsColumns.SPLIT] = Split.TEST
    neg_train[CropsColumns.SPLIT] = Split.TRAIN
    neg_val[CropsColumns.SPLIT] = Split.VAL
    neg_test[CropsColumns.SPLIT] = Split.TEST

    df_with_splits = pd.concat((pos_train, neg_train, pos_val, neg_val, pos_test, neg_test), axis=0)
    df_with_splits.to_csv(crops_dataset, header=True, index=False)


def create_contrastive_crops_dataset(crops_folder: str, coords_csv: str, crops_dataset: str,
                                     show_sampling_result: bool = False, contrastive_samples_percent: float = 0.25,
                                     contrastive_distance_multiplier: float = 1.1, pos_data_upsampling: bool = False,
                                     pos_upsample_dist: int = 3, neg_upsample_multiplier: float = 0):
    global plt # don't ask why.
    if not os.path.exists(crops_folder):
        os.makedirs(crops_folder)

    crop_name_list = []
    orig_name_list = []
    x_list = []
    y_list = []
    label_list = []

    n_positives = 0
    label = 1
    dataset = CoordinatesDataset(coords_csv)
    print('Creating positive crops...')
    firstx, firsty = True, True
    for data_filename, label_filename in dataset.iterate_data(Split.TRAIN):
        if label_filename is None:
            continue
        print(data_filename)
        orig_img_name = os.path.basename(data_filename)
        img_name = os.path.splitext(orig_img_name)[0]

        img = open_image(data_filename)
        coordinates = dataset.load_coordinates(label_filename)

        for x_center, y_center in coordinates:
            crop = create_crop(img, x_center, y_center)
            crop_name = "{}_{}_{}.tif".format(img_name, x_center, y_center)
            crop.save(os.path.join(crops_folder, crop_name))
            if firstx:
                firstx = False
                crop_save(crop, "pos.png")
                print('saved')

            crop_name_list.append(crop_name)
            orig_name_list.append(orig_img_name)
            x_list.append(x_center)
            y_list.append(y_center)
            label_list.append(label)
            if pos_data_upsampling:
                x_rand, y_rand = None, None
                while x_rand is None:
                    rand_angle = np.random.uniform(0, 2 * np.pi)
                    x_rand = round(pos_upsample_dist * np.cos(rand_angle)) + x_center
                    y_rand = round(pos_upsample_dist * np.sin(rand_angle)) + y_center
                    out_of_bounds = x_rand >= img.size[0] or y_rand >= img.size[1] or \
                                    x_rand < 0 or y_rand < 0
                    if out_of_bounds != 0:
                        x_rand, y_rand = None, None

                crop = create_crop(img, x_rand, y_rand)
                crop_name = "{}_{}_{}.tif".format(img_name, x_rand, y_rand)
                crop.save(os.path.join(crops_folder, crop_name))
                crop_name_list.append(crop_name)
                orig_name_list.append(orig_img_name)
                x_list.append(x_center)
                y_list.append(y_center)
                label_list.append(label)

                if firsty:
                    firsty = False
                    crop_save(crop, "pos_jit.png")

            n_positives += 1

    label = 0
    no_train_images = dataset.split_length(Split.TRAIN)
    contrastive_sampling_distance = (window_size[0] * contrastive_distance_multiplier) // 2
    neg_crops_per_image = [round((n_positives // no_train_images) * (1+neg_upsample_multiplier)) + (1 if x < n_positives % no_train_images else 0) for x in
                           range(no_train_images)]
    neg_non_constrastive_crops_per_image, neg_contrastive_crops_per_image = \
        list(zip(*[(n_crops - round(contrastive_samples_percent * n_crops),
                   round(contrastive_samples_percent * n_crops))
                   for n_crops in neg_crops_per_image]))
    firstx, firsty = True, True
    # neg_non_constrastive_crops_per_image, neg_contrastive_crops_per_image = 30*[0], 30*[44]
    print(contrastive_sampling_distance)
    print('Creating contrastive negative crops...')
    for (data_filename, label_filename), no_neg_crops in zip(dataset.iterate_data(Split.TRAIN),
                                                             neg_contrastive_crops_per_image):
        print(data_filename)
        orig_img_name = os.path.basename(data_filename)
        img_name = os.path.splitext(orig_img_name)[0]
        img = open_image(data_filename)

        if label_filename:
            coordinates = dataset.load_coordinates(label_filename)
            support_map = generate_support_img(coordinates, window_size)
        else:
            support_map = None

        for idx in np.random.choice(len(coordinates), no_neg_crops):
            atom_rand = coordinates[idx]
            x_center, y_center = atom_rand
            x_rand, y_rand = None, None
            if support_map is not None:
                retries=0
                while x_rand is None and retries < 50:  # Extremely unlikely: sample impossible
                    retries += 1
                    rand_angle = np.random.uniform(0, 2 * np.pi)
                    x_rand = round(contrastive_sampling_distance * np.cos(rand_angle)) + x_center
                    y_rand = round(contrastive_sampling_distance * np.sin(rand_angle)) + y_center
                    out_of_bounds = x_rand >= img.size[0] or y_rand >= img.size[1] or \
                        x_rand<0 or y_rand<0
                    if out_of_bounds or support_map[x_rand, y_rand] != 0:
                        x_rand, y_rand = None, None

            x_center, y_center = x_rand, y_rand

            crop = create_crop(img, x_center, y_center)
            crop_name = "{}_{}_{}.tif".format(img_name, x_center, y_center)
            crop.save(os.path.join(crops_folder, crop_name))

            crop_name_list.append(crop_name)
            orig_name_list.append(orig_img_name)
            x_list.append(x_center)
            y_list.append(y_center)
            label_list.append(label)
            if firsty:
                firsty = False
                crop_save(crop, "neg_con.png")

    print('Creating non-contrastive negative crops...')
    for (data_filename, label_filename), no_neg_crops in zip(dataset.iterate_data(Split.TRAIN),
                                                             neg_non_constrastive_crops_per_image):
        print(data_filename)
        orig_img_name = os.path.basename(data_filename)
        img_name = os.path.splitext(orig_img_name)[0]
        img = open_image(data_filename)

        if label_filename:
            coordinates = dataset.load_coordinates(label_filename)
            support_map = generate_support_img(coordinates, window_size)
        else:
            support_map = None

        for _ in range(no_neg_crops):
            x_rand = np.random.randint(0, 512)
            y_rand = np.random.randint(0, 512)

            if support_map is not None:
                while support_map[x_rand, y_rand] != 0:
                    x_rand = np.random.randint(0, 512)
                    y_rand = np.random.randint(0, 512)

            x_center, y_center = x_rand, y_rand

            crop = create_crop(img, x_center, y_center)
            crop_name = "{}_{}_{}.tif".format(img_name, x_center, y_center)
            crop.save(os.path.join(crops_folder, crop_name))

            crop_name_list.append(crop_name)
            orig_name_list.append(orig_img_name)
            x_list.append(x_center)
            y_list.append(y_center)
            label_list.append(label)
            if firstx:
                firstx = False
                crop_save(crop, "neg_ncon.png")

    if show_sampling_result:
        # Only works for single img data.
        positives = [(x, y) for x,y,l in zip(x_list, y_list, label_list) if l==1]
        negatives = [(x, y) for x,y,l in zip(x_list, y_list, label_list) if l==0]
        from matplotlib import pyplot as plt
        plt.imshow(img)
        plt.scatter(*zip(*positives))
        plt.scatter(*zip(*negatives))
        plt.show()




    df_data = {
        CropsColumns.FILENAME: crop_name_list,
        CropsColumns.ORIGINAL: orig_name_list,
        CropsColumns.X: x_list,
        CropsColumns.Y: y_list,
        CropsColumns.LABEL: label_list
    }
    df = pd.DataFrame(df_data, columns=[
        CropsColumns.FILENAME,
        CropsColumns.ORIGINAL,
        CropsColumns.X,
        CropsColumns.Y,
        CropsColumns.LABEL
    ])

    df_pos = df[df.Label == 1]
    df_neg = df[df.Label == 0]

    pos_len = len(df_pos)
    neg_len = len(df_neg)

    pos_train, pos_val = np.split(df_pos.sample(frac=1), [int(0.9 * pos_len)])
    neg_train, neg_val = np.split(df_neg.sample(frac=1), [int(0.9 * neg_len)])
    pos_train[CropsColumns.SPLIT] = Split.TRAIN
    pos_val[CropsColumns.SPLIT] = Split.VAL
    neg_train[CropsColumns.SPLIT] = Split.TRAIN
    neg_val[CropsColumns.SPLIT] = Split.VAL
    print("Final size for train(P vs N):", len(pos_train), len(neg_train))
    print("Final size for val (P vs N):", len(pos_val), len(neg_val))
    df_with_splits = pd.concat((pos_train, neg_train, pos_val, neg_val), axis=0)
    df_with_splits.to_csv(crops_dataset, header=True, index=False)


def crop_save(crop, im_name):
    crop = np.array(crop)
    crop = (crop + crop.min()) * 500
    crop = Image.fromarray(crop)
    crop = crop.convert("L")
    crop.save(im_name, 'png')


if __name__ == "__main__":
    create_crops_dataset(CROPS_PATH, PT_DATASET, CROPS_DATASET)
