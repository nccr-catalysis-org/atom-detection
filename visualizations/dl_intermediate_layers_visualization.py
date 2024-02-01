import os
from typing import List, Tuple, Optional, Dict

import argparse

from PIL import Image
import numpy as np
import torch
import torch.nn.functional
from matplotlib import pyplot as plt

from atoms_detection.dataset import CoordinatesDataset
from atoms_detection.image_preprocessing import dl_prepro_image
from atoms_detection.model import BasicCNN
from utils.constants import ModelArgs, Split
from utils.paths import ACTIVATIONS_VIS_PATH


class ConvLayerVisualizer:
    CONV_0 = 'Conv0'
    CONV_3 = 'Conv3'
    CONV_6 = 'Conv6'

    def __init__(self, model_name: ModelArgs, ckpt_filename: str):
        self.model_name = model_name
        self.ckpt_filename = ckpt_filename
        self.device = self.get_torch_device()
        self.batch_size = 64

        self.stride = 1
        self.padding = 10
        self.window_size = (21, 21)

    @staticmethod
    def get_torch_device():
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def sliding_window(self, image: np.ndarray) -> Tuple[int, int, np.ndarray]:
        # slide a window across the image
        x_to_center = self.window_size[0] // 2 - 1 if self.window_size[0] % 2 == 0 else self.window_size[0] // 2
        y_to_center = self.window_size[1] // 2 - 1 if self.window_size[1] % 2 == 0 else self.window_size[1] // 2

        for y in range(0, image.shape[0] - self.window_size[1]+1, self.stride):
            for x in range(0, image.shape[1] - self.window_size[0]+1, self.stride):
                # yield the current window
                center_x = x + x_to_center
                center_y = y + y_to_center
                yield center_x, center_y, image[y:y + self.window_size[1], x:x + self.window_size[0]]

    def padding_image(self, img: np.ndarray) -> np.ndarray:
        image_padded = np.zeros((img.shape[0] + self.padding*2, img.shape[1] + self.padding*2))
        image_padded[self.padding:-self.padding, self.padding:-self.padding] = img
        return image_padded

    def images_to_torch_input(self, image: np.ndarray) -> torch.Tensor:
        expanded_img = np.expand_dims(image, axis=(0, 1))
        input_tensor = torch.from_numpy(expanded_img).float()
        input_tensor = input_tensor.to(self.device)
        return input_tensor

    def load_model(self) -> BasicCNN:
        checkpoint = torch.load(self.ckpt_filename, map_location=self.device)
        model = BasicCNN(num_classes=2).to(self.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model

    @staticmethod
    def center_to_slice(x_center: int, y_center: int, width: int, height: int) -> Tuple[slice, slice]:
        x_to_center = width // 2 - 1 if width % 2 == 0 else width // 2
        y_to_center = height // 2 - 1 if height % 2 == 0 else height // 2
        x = x_center - x_to_center
        y = y_center - y_to_center
        return slice(x, x + width), slice(y, y + height)

    def get_prediction_map(self, padded_image: np.ndarray) -> Dict[str, np.ndarray]:
        _shape = padded_image.shape
        convs_activations_dict = {
            self.CONV_0: (np.zeros(_shape), np.zeros(_shape)),
            self.CONV_3: (np.zeros(_shape), np.zeros(_shape)),
            self.CONV_6: (np.zeros(_shape), np.zeros(_shape))
        }
        model = self.load_model()
        for x, y, image_crop in self.sliding_window(padded_image):
            torch_input = self.images_to_torch_input(image_crop)
            conv_outputs = self.get_conv_activations(torch_input, model)
            for conv_layer_key, activations_blob in conv_outputs.items():
                activation_map = self.sum_channels(activations_blob)
                h, w = activation_map.shape
                x_slice, y_slice = self.center_to_slice(x, y, w, h)
                convs_activations_dict[conv_layer_key][0][y_slice, x_slice] += 1
                convs_activations_dict[conv_layer_key][1][y_slice, x_slice] += activation_map

        activations_dict = {}
        for conv_layer_key, (counting_map, output_map) in convs_activations_dict.items():
            zero_rows = np.sum(counting_map, axis=1)
            zero_cols = np.sum(counting_map, axis=0)

            output_map = np.delete(output_map, np.where(zero_rows == 0), axis=0)
            clean_output_map = np.delete(output_map, np.where(zero_cols == 0), axis=1)
            counting_map = np.delete(counting_map, np.where(zero_rows == 0), axis=0)
            clean_counting_map = np.delete(counting_map, np.where(zero_cols == 0), axis=1)

            activations_dict[conv_layer_key] = clean_output_map / clean_counting_map

        return activations_dict

    def get_conv_activations(self, input_image: torch.Tensor, model: BasicCNN) -> Dict[str, np.ndarray]:
        conv_activations = {}
        activations = input_image
        for i, layer in enumerate(model.features):
            activations = layer(activations)
            if i == 0:
                conv_activations[self.CONV_0] = activations.squeeze(0).detach().cpu().numpy()
            elif i == 3:
                conv_activations[self.CONV_3] = activations.squeeze(0).detach().cpu().numpy()
            elif i == 6:
                conv_activations[self.CONV_6] = activations.squeeze(0).detach().cpu().numpy()

        return conv_activations

    @staticmethod
    def sum_channels(activations: np.ndarray):
        aggregated_activations = np.sum(activations, axis=0)
        return aggregated_activations

    def image_to_pred_map(self, img: np.ndarray) -> Dict[str, np.ndarray]:
        preprocessed_img = dl_prepro_image(img)
        padded_image = self.padding_image(preprocessed_img)
        activations_dict = self.get_prediction_map(padded_image)
        return activations_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "architecture",
        type=ModelArgs,
        choices=ModelArgs,
        help="Architecture name"
    )
    parser.add_argument(
        "ckpt_filename",
        type=str,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "coords_csv",
        type=str,
        help="Coordinates CSV file to use as input"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print(args)

    conv_visualizer = ConvLayerVisualizer(
        model_name=args.architecture,
        ckpt_filename=args.ckpt_filename
    )

    coordinates_dataset = CoordinatesDataset(args.coords_csv)
    for image_path, coordinates_path in coordinates_dataset.iterate_data(Split.TEST):
        img = Image.open(image_path)
        np_img = np.array(img)
        activations_dict = conv_visualizer.image_to_pred_map(np_img)

        img_name = os.path.splitext(os.path.basename(image_path))[0]

        output_folder = os.path.join(ACTIVATIONS_VIS_PATH, f"{img_name}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for conv_layer_key, activation_map in activations_dict.items():
            fig = plt.figure()
            plt.title(f"{conv_layer_key} -- {img_name}")
            plt.imshow(activation_map)

            output_path = os.path.join(output_folder, f"{conv_layer_key}_{img_name}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)



