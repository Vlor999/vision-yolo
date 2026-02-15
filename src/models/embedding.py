"""Embedding utilities for object matching and image processing."""

import os
from collections.abc import Generator
from typing import TypeAlias

import cv2 as cv
import numpy as np
from loguru import logger
from PIL import Image, UnidentifiedImageError

# =============================================================================
# Type Aliases
# =============================================================================

Points: TypeAlias = list[tuple[int, int]]
EmbeddingEntry: TypeAlias = tuple[np.ndarray, str]
EmbeddingDict: TypeAlias = dict[str, list[EmbeddingEntry]]

# Default size to which crops are resized before computing the embedding vector
DEFAULT_EMBEDDING_SIZE: tuple[int, int] = (64, 64)


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


# =============================================================================
# Embedding Vector Computation
# =============================================================================


def compute_embedding_vector(
    crop: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_EMBEDDING_SIZE,
) -> np.ndarray:
    """Compute a normalised embedding vector from an image crop.

    The crop is resized to *target_size*, converted to ``float32``,
    flattened into a 1-D vector, and L2-normalised so that cosine
    similarity can be computed with a simple dot product.

    Args:
        crop: Image region as a numpy array (H x W x C).
        target_size: ``(width, height)`` to resize the crop to before
            flattening.  A smaller size is faster but less discriminative.

    Returns:
        A 1-D ``float32`` vector of length ``width * height * channels``
        with unit L2 norm.

    Raises:
        ValueError: If *crop* is empty or has fewer than 2 dimensions.
    """
    if crop.size == 0 or crop.ndim < 2:
        raise ValueError("crop must be a non-empty image array (H x W [x C])")

    resized = cv.resize(crop, target_size, interpolation=cv.INTER_LINEAR)
    vector = resized.astype(np.float32).flatten()
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


# =============================================================================
# Embedding Update Helpers
# =============================================================================


def update_embedding(
    embedding: EmbeddingDict,
    category: str,
    label: str,
    image: np.ndarray,
    box: Points | None = None,
    target_size: tuple[int, int] = DEFAULT_EMBEDDING_SIZE,
) -> EmbeddingDict:
    """Add a new reference embedding for *label* under *category*.

    If *box* is provided, the region is cropped from *image* first;
    otherwise *image* is used as-is (assumed to already be a crop of
    the subject).

    Args:
        embedding: The mutable embedding dictionary to update.
        category: Detection category key (e.g. ``"person"``).
        label: Human-readable identifier (e.g. ``"willem"``).
        image: Full image or pre-cropped region (numpy array).
        box: Optional bounding-box points ``[(x0, y0), (x1, y1)]``
            used to crop *image*.  For an axis-aligned box only the
            first two points are used.
        target_size: Size passed to :func:`compute_embedding_vector`.

    Returns:
        The updated *embedding* dictionary (same object, mutated in
        place for convenience).
    """
    if box is not None:
        p0, p1 = box[0], box[1]
        x0, y0 = p0
        x1, y1 = p1
        crop = image[y0:y1, x0:x1]
    else:
        crop = image

    if crop.size == 0:
        logger.warning(
            f"Empty crop for label '{label}' in category '{category}' skipped"
        )
        return embedding

    vector = compute_embedding_vector(crop, target_size=target_size)
    entry: EmbeddingEntry = (vector, label)

    if category not in embedding:
        embedding[category] = []

    embedding[category].append(entry)
    logger.info(
        f"Embedding updated: category='{category}', label='{label}', "
        f"vector_size={vector.shape[0]}"
    )
    return embedding


def build_embedding_from_directory(
    directory: str = "photos",
    category: str = "person",
    target_size: tuple[int, int] = DEFAULT_EMBEDDING_SIZE,
) -> EmbeddingDict:
    """Build an embedding dictionary from a directory of reference images.

    The expected layout is::

        directory/
            label_a/
                img1.jpg
                img2.jpg
            label_b/
                img3.png

    Each sub-directory name is used as the *label* for every image it
    contains.  Images directly in *directory* (not in a sub-directory)
    are ignored.

    Args:
        directory: Root directory to scan.
        category: Detection category under which all entries are stored
            (e.g. ``"person"``).
        target_size: Size passed to :func:`compute_embedding_vector`.

    Returns:
        A new :data:`EmbeddingDict` ready to be passed to
        :func:`match_boxes_embedding`.
    """
    embedding: EmbeddingDict = {}

    if not os.path.isdir(directory):
        logger.error(f"Directory not found: {directory}")
        return embedding

    for label in sorted(os.listdir(directory)):
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue

        image_files = found_all_images(label_dir, verbose=False)
        for filepath in image_files:
            try:
                image = np.array(Image.open(filepath))
                update_embedding(
                    embedding,
                    category=category,
                    label=label,
                    image=image,
                    target_size=target_size,
                )
            except (UnidentifiedImageError, ValueError) as exc:
                logger.error(f"Skipping {filepath}: {exc}")

    logger.info(
        f"Built embedding from '{directory}': "
        f"{sum(len(v) for v in embedding.values())} vectors across "
        f"{len(embedding)} categories"
    )
    return embedding


# =============================================================================
# Embedding Matching
# =============================================================================


def match_boxes_embedding(
    boxes_per_categorie: dict[str, list[tuple[list[tuple[int, int]], float]]],
    embedding: EmbeddingDict,
    matrix: np.ndarray,
    threshold: float | int = 0.90,
    target_size: tuple[int, int] = DEFAULT_EMBEDDING_SIZE,
) -> dict[str, list[tuple[list[tuple[int, int]], float, str | None]]]:
    """Match detected boxes with stored reference embeddings.

    For every detection whose category exists in *embedding*, a crop of
    the detected region is turned into an embedding vector and compared
    (cosine similarity) against all stored reference vectors.  When the
    best similarity exceeds *threshold*, the matched label is attached
    to the detection.

    Args:
        boxes_per_categorie: ``{category: [(points, confidence), …]}``.
        embedding: Reference embeddings built via
            :func:`update_embedding` or
            :func:`build_embedding_from_directory`.
        matrix: The full image frame (numpy array, H x W x C).
        threshold: Minimum cosine similarity to accept a match.
        target_size: Size used when computing the embedding vector for
            each detected crop (must match the size used when the
            reference was created).

    Returns:
        A new dictionary with the same structure as *boxes_per_categorie*
        but each entry is extended to ``(points, confidence, label)``
        where *label* is ``None`` when no match was found.
    """
    result: dict[str, list[tuple[list[tuple[int, int]], float, str | None]]] = {}

    for cat, detections in boxes_per_categorie.items():
        result[cat] = []

        if cat not in embedding or len(embedding[cat]) == 0:
            for points, conf in detections:
                result[cat].append((points, conf, None))
            continue

        ref_vectors = np.array([entry[0] for entry in embedding[cat]])
        ref_labels = [entry[1] for entry in embedding[cat]]

        for points, conf in detections:
            p0, p1 = points[0], points[1]
            x0, y0 = p0
            x1, y1 = p1
            crop = matrix[y0:y1, x0:x1]

            if crop.size == 0:
                result[cat].append((points, conf, None))
                continue

            det_vector = compute_embedding_vector(crop, target_size=target_size)

            similarities = ref_vectors @ det_vector
            best_idx = int(np.argmax(similarities))
            best_sim = float(similarities[best_idx])

            if best_sim >= threshold:
                matched_label = ref_labels[best_idx]
                logger.debug(
                    f"Matched {cat} → '{matched_label}' (similarity={best_sim:.3f})"
                )
                result[cat].append((points, conf, matched_label))
            else:
                result[cat].append((points, conf, None))

    return result


if __name__ == "__main__":
    it_images = iterator_images()
    # for image in it_images:
    #     print(image)
