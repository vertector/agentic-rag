"""
Visualization utilities for document parsing results.

This module provides helper functions for visualizing layout detection results,
displaying document pages, and cropping regions from images.
"""

import base64
from io import BytesIO
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image

from shared.schemas import Document


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Decode a base64 string to a numpy array image.

    Args:
        base64_string: Base64 encoded PNG image string.

    Returns:
        Numpy array of the image (BGR format for OpenCV compatibility).
    """
    image_data = base64.b64decode(base64_string)
    pil_image = Image.open(BytesIO(image_data))
    return np.array(pil_image)


def crop_region(image, bbox: List[int], padding: int = 0) -> Image.Image:
    """
    Crop a region from an image with optional padding.

    Args:
        image: PIL Image or numpy array.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        padding: Optional padding in pixels to add around the region.

    Returns:
        Cropped PIL Image.
    """
    x1, y1, x2, y2 = bbox

    if not hasattr(image, "width"):
        image = Image.fromarray(image)

    # Apply padding with bounds checking
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)

    return image.crop((x1, y1, x2, y2))


def visualize_layout(
    document: Document,
    min_confidence: float = 0.5,
    colormap_name: str = "tab20",
) -> np.ndarray:
    """
    Visualize layout detection results by drawing bounding boxes on the page image.

    Args:
        document: Document object containing chunks with grounding info.
        min_confidence: Minimum confidence threshold to display a chunk.
        colormap_name: Matplotlib colormap name for generating colors.

    Returns:
        Numpy array of the image with bounding boxes drawn.
    """
    img_plot = base64_to_image(document.metadata.page_image_base64).copy()

    # Get all unique labels
    labels = list(set(chunk.grounding.chunk_type for chunk in document.chunks))

    # Generate colors dynamically from colormap
    cmap = colormaps.get_cmap(colormap_name)
    color_map = {}
    for i, label in enumerate(labels):
        rgba = cmap(i % 20)
        # Convert to BGR (0-255) for OpenCV
        color_map[label] = (int(rgba[2] * 255), int(rgba[1] * 255), int(rgba[0] * 255))

    for chunk in document.chunks:
        if chunk.grounding.score < min_confidence:
            continue

        label = chunk.grounding.chunk_type
        score = chunk.grounding.score
        coords = chunk.grounding.bbox
        color = color_map[label]

        x1, y1, x2, y2 = [int(c) for c in coords]
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)

        cv2.polylines(img_plot, [pts], True, color, 2)
        text = f"{label} ({score:.2f})"
        cv2.putText(img_plot, text, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_plot


def display_layout(
    document: Document,
    min_confidence: float = 0.5,
    figsize: tuple = (12, 14),
    title: str = "Layout Detection Results",
) -> None:
    """
    Display the layout detection results using matplotlib.

    Args:
        document: Document object containing chunks with grounding info.
        min_confidence: Minimum confidence threshold to display.
        figsize: Figure size as (width, height) tuple.
        title: Plot title.
    """
    result_image = visualize_layout(
        document, 
        min_confidence
        )

    plt.figure(figsize=figsize)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
