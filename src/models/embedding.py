"""Embedding utilities for object matching and image processing."""

import os
from collections.abc import Generator
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError


def iterator_images_from_filenames(
    filenames: list[str],
) -> Generator[np.ndarray, None, None]:
    """Yield images as numpy arrays from a list of filenames.

    Args:
        filenames: List of image file paths to load.

    Yields:
        Numpy array representation of each image.
    """
    for filename in filenames:
        try:
            image = np.array(Image.open(filename))
            yield np.array(image)
        except UnidentifiedImageError as image_error:
            logger.error(image_error)


def found_all_images(directory: str = "photos", *, verbose: bool = True) -> list[str]:
    """Find all image files in a directory.

    Args:
        directory: Directory path to search for images.
        verbose: Whether to log found files.

    Returns:
        List of absolute paths to found image files.
    """
    try:
        files = []
        for f in os.listdir(directory):
            joined_path = os.path.join(directory, f)
            if os.path.isfile(joined_path):
                absolute_path = os.path.abspath(joined_path)
                if verbose:
                    logger.info(f"File found: {absolute_path}")
                files.append(absolute_path)

        return files
    except Exception as e:
        logger.error(e)
        return []


def iterator_images(
    directory: str = "photos",
) -> Generator[np.ndarray, None, None]:
    """Yield images from a directory as numpy arrays.

    Args:
        directory: Directory path containing images.

    Yields:
        Numpy array representation of each image.
    """
    files = found_all_images(directory=directory)
    return iterator_images_from_filenames(files)


def match_boxes_embedding(
    boxes_per_categorie: dict[str, list[tuple[list[tuple[int, int]], float]]],
    embedding: dict[Any, Any],
) -> dict[str, list[tuple[list[tuple[int, int]], float]]]:
    """Match detected boxes with embedding vectors for object recognition.

    Args:
        boxes_per_categorie: Dictionary mapping category IDs to lists of
            bounding boxes. Each box is (points, confidence).
        embedding: Dictionary mapping category IDs to embedding data.

    Returns:
        The input boxes_per_categorie dictionary (unchanged placeholder).

    Note:
        This function currently just logs the embedding information
        and returns the input unchanged. Placeholder for future use.
    """
    for categorie, vect in embedding.items():
        logger.debug(f"Category {categorie}: {len(vect)} embeddings")
        for i, (vector, name) in enumerate(vect):
            logger.debug(f"{i}, {vector, name}")

    return boxes_per_categorie


if __name__ == "__main__":
    it_images = iterator_images()
    # for image in it_images:
    #     print(image)
