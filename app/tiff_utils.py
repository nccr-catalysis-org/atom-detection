#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author : Romain Graux
@date : 2023 April 25, 11:59:06
@last modified : 2023 September 19, 11:18:36
"""

from typing import Callable, Optional
import re
import imageio
from collections import namedtuple
import numpy as np

PhysicalMetadata = namedtuple(
    "PhysicalMetadata", ["width", "height", "pixel_width", "pixel_height", "unit"]
)

MetadataExtractor = Callable[[dict, int, int], Optional[PhysicalMetadata]]


def extract_imagej_metadata(
    metadata: dict, width: int, height: int
) -> Optional[PhysicalMetadata]:
    try:
        ipw, iph, _ = metadata["resolution"]
        result = re.search(r"unit=(.+)", metadata["description"])
        if not result:
            return None
        unit = result.group(1)
        return PhysicalMetadata(width, height, 1.0 / ipw, 1.0 / iph, unit.lower())
    except (KeyError, AttributeError):
        return None


def extract_resolution_metadata(
    metadata: dict, width: int, height: int
) -> Optional[PhysicalMetadata]:
    try:
        ipw, iph, _ = metadata["resolution"]
        # It looks like the resolution unit is not really reliable, so let's just assume nm
        unit = "nm"
        return PhysicalMetadata(width, height, 1.0 / ipw, 1.0 / iph, unit)
    except (KeyError, AttributeError):
        return None


METADATA_EXTRACTORS: list[MetadataExtractor] = [
    extract_imagej_metadata,
    extract_resolution_metadata,
]


def normalize_metadata(metadata: PhysicalMetadata) -> PhysicalMetadata:
    conversion_factor = {
        "inch": 2.54e7,
        "m": 1e9,
        "dm": 1e8,
        "cm": 1e7,
        "mm": 1e6,
        "µm": 1e3,
        "nm": 1,
    }
    if metadata.unit not in conversion_factor:
        raise ValueError(f"Unknown unit: {metadata.unit}")
    factor = conversion_factor[metadata.unit]
    return PhysicalMetadata(
        metadata.width,
        metadata.height,
        metadata.pixel_width * factor,
        metadata.pixel_height * factor,
        "nm",
    )


def extract_physical_metadata(image_path: str, strict: bool = True) -> PhysicalMetadata:
    """
    Extracts the physical metadata of an image by trying all available extractors.
    Raises ValueError if no extractor succeeds.
    """
    with open(image_path, "rb") as f:
        data = f.read()
        reader = imageio.get_reader(data)
        metadata = reader.get_meta_data()

    h, w = reader.get_next_data().shape
    for extractor in METADATA_EXTRACTORS:
        result = extractor(metadata, w, h)
        if result is not None:
            return normalize_metadata(result)

    raise ValueError(
        "Failed to extract metadata from the image using any available method."
    )


def tiff_to_png(image, inplace=True):
    img = image if inplace else image.copy()
    if np.array(img.getdata()).max() <= 1:
        img = img.point(lambda p: p * 255)
    return img.convert("RGB")
