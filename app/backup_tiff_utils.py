#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 April 25, 11:59:06
@last modified : 2023 June 20, 15:04:37
"""

import re
import imageio
import numpy as np
from collections import namedtuple
from typing import Protocol

physical_metadata = namedtuple("physical_metadata", ["width", "height", "pixel_width", "pixel_height", "unit"])


class ImageMetadataExtractor(Protocol):
    @classmethod
    def __call__(cls, image_path:str, strict:bool=True) -> physical_metadata:
        ...

class TIFFMetadataExtractor(ImageMetadataExtractor):
    @classmethod
    def __call__(cls, image_path:str, strict:bool=True) -> physical_metadata:
        """
        Extracts the physical metadata of an image (only tiff for now)
        """
        with open(image_path, "rb") as f:
            data = f.read()
            reader = imageio.get_reader(data, format=".tif")
            metadata = reader.get_meta_data()

        if strict and not metadata['is_imagej']:
            for key, value in metadata.items():
                if key.startswith("is_") and value == True: # Force bool to be True, because it can also pass the condition while being an random object
                    raise ValueError(f"The image is not TIFF image, but it seems to be a {key[3:]} image")
            raise ValueError("The image is not in TIFF format")
        h, w = reader.get_next_data().shape
        ipw, iph, _ = metadata['resolution']
        result = re.search(r"unit=(.+)", metadata['description'])
        if strict and not result:
            raise ValueError(f"No scale unit found in the image description : {metadata['description']}")
        unit = result and result.group(1)
        return physical_metadata(w, h, 1. / ipw, 1. / iph, unit)

def extract_physical_metadata(image_path : str, strict:bool=True) -> physical_metadata:
    if image_path.endswith(".tif"):
        return TIFFMetadataExtractor(image_path, strict)

def tiff_to_png(image, inplace=True):
    img = image if inplace else image.copy()
    if np.array(img.getdata()).max() <= 1:
        img = img.point(lambda p: p * 255)
    return img.convert("RGB")
