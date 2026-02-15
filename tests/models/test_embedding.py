"""Tests for `models/embedding.py`."""

import numpy as np
import pytest

from src.models.embedding import (
    build_embedding_from_directory,
    compute_embedding_vector,
    match_boxes_embedding,
    update_embedding,
)

# =============================================================================
# compute_embedding_vector
# =============================================================================


class TestComputeEmbeddingVector:
    """Tests for compute_embedding_vector."""

    def test_returns_1d_vector(self) -> None:
        crop = np.random.randint(0, 255, (30, 40, 3), dtype=np.uint8)
        vec = compute_embedding_vector(crop, target_size=(16, 16))
        assert vec.ndim == 1
        assert vec.shape == (16 * 16 * 3,)

    def test_unit_norm(self) -> None:
        crop = np.random.randint(1, 255, (20, 20, 3), dtype=np.uint8)
        vec = compute_embedding_vector(crop)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_zero_image_returns_zero_vector(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        vec = compute_embedding_vector(crop)
        assert np.allclose(vec, 0.0)

    def test_deterministic(self) -> None:
        crop = np.random.randint(0, 255, (25, 25, 3), dtype=np.uint8)
        v1 = compute_embedding_vector(crop)
        v2 = compute_embedding_vector(crop)
        np.testing.assert_array_equal(v1, v2)

    def test_raises_on_empty_crop(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_embedding_vector(np.array([]))

    def test_raises_on_1d_input(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compute_embedding_vector(np.array([1, 2, 3]))


# =============================================================================
# update_embedding
# =============================================================================


class TestUpdateEmbedding:
    """Tests for update_embedding."""

    def test_creates_category_if_missing(self) -> None:
        emb: dict = {}
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        result = update_embedding(emb, "person", "alice", image)
        assert "person" in result
        assert len(result["person"]) == 1
        assert result["person"][0][1] == "alice"

    def test_appends_to_existing_category(self) -> None:
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        emb: dict = {}
        update_embedding(emb, "person", "alice", image)
        update_embedding(emb, "person", "bob", image)
        assert len(emb["person"]) == 2
        assert emb["person"][1][1] == "bob"

    def test_with_bounding_box_crop(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb: dict = {}
        box = [(10, 20), (60, 80)]
        update_embedding(emb, "person", "alice", image, box=box)
        assert "person" in emb
        assert len(emb["person"]) == 1

    def test_empty_crop_skipped(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb: dict = {}
        # box with zero area
        box = [(50, 50), (50, 50)]
        update_embedding(emb, "person", "ghost", image, box=box)
        assert "person" not in emb

    def test_mutates_in_place(self) -> None:
        emb: dict = {}
        image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        result = update_embedding(emb, "person", "alice", image)
        assert result is emb


# =============================================================================
# build_embedding_from_directory
# =============================================================================


class TestBuildEmbeddingFromDirectory:
    """Tests for build_embedding_from_directory."""

    def test_builds_from_subdirectories(self, tmp_path) -> None:
        """Create a temp dir structure with fake images and build embeddings."""
        from PIL import Image

        label_dir = tmp_path / "alice"
        label_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8))
        img.save(label_dir / "photo1.png")
        img.save(label_dir / "photo2.png")

        emb = build_embedding_from_directory(
            str(tmp_path), category="person", target_size=(16, 16)
        )
        assert "person" in emb
        assert len(emb["person"]) == 2
        assert all(entry[1] == "alice" for entry in emb["person"])

    def test_ignores_files_in_root(self, tmp_path) -> None:
        from PIL import Image

        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        img.save(tmp_path / "root_image.png")

        emb = build_embedding_from_directory(str(tmp_path), category="person")
        assert len(emb) == 0

    def test_missing_directory_returns_empty(self) -> None:
        emb = build_embedding_from_directory("/nonexistent/path")
        assert emb == {}


# =============================================================================
# match_boxes_embedding
# =============================================================================


class TestMatchBoxesEmbedding:
    """Tests for match_boxes_embedding."""

    @staticmethod
    def _make_embedding_from_crop(
        crop: np.ndarray,
        label: str,
        target_size: tuple[int, int] = (16, 16),
    ) -> dict:
        vec = compute_embedding_vector(crop, target_size=target_size)
        return {"person": [(vec, label)]}

    def test_matches_identical_crop(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = image[20:60, 10:50]
        emb = self._make_embedding_from_crop(crop, "alice", target_size=(16, 16))

        boxes = {"person": [([(10, 20), (50, 60)], 0.95)]}
        result = match_boxes_embedding(
            boxes, emb, image, threshold=0.9, target_size=(16, 16)
        )

        assert result["person"][0][2] == "alice"

    def test_no_match_below_threshold(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Embedding from a completely different crop
        other = np.random.randint(0, 255, (40, 40, 3), dtype=np.uint8) + 128
        emb = self._make_embedding_from_crop(other, "bob", target_size=(16, 16))

        boxes = {"person": [([(10, 20), (50, 60)], 0.8)]}
        result = match_boxes_embedding(
            boxes, emb, image, threshold=0.999, target_size=(16, 16)
        )

        assert result["person"][0][2] is None

    def test_category_without_embedding_gets_none(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb: dict = {}

        boxes = {"car": [([(0, 0), (50, 50)], 0.9)]}
        result = match_boxes_embedding(boxes, emb, image)

        assert result["car"][0][2] is None

    def test_preserves_points_and_confidence(self) -> None:
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        emb: dict = {}
        pts = [(5, 10), (45, 55)]
        boxes = {"dog": [(pts, 0.77)]}
        result = match_boxes_embedding(boxes, emb, image)

        assert result["dog"][0][0] == pts
        assert result["dog"][0][1] == 0.77
