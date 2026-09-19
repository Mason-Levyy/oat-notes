import numpy as np
import pytest

from oat_notes.embeddings import cosine, cosine_scores, normalize


def test_normalize_returns_a_float32_unit_vector():
    vector = normalize(np.array([[3.0, 4.0]], dtype=np.float64))
    assert vector.dtype == np.float32
    assert vector.shape == (2,)
    assert np.allclose(vector, [0.6, 0.8])
    assert vector.flags["C_CONTIGUOUS"]


@pytest.mark.parametrize("bad", [[0.0, 0.0], [np.nan, 1.0], [np.inf, 0.0]])
def test_normalize_refuses_directionless_vectors(bad):
    with pytest.raises(ValueError, match="finite non-zero norm"):
        normalize(np.array(bad, dtype=np.float32))


def test_cosine_is_none_across_shapes():
    unit = np.array([1.0, 0.0], dtype=np.float32)
    assert cosine(unit, np.array([1.0, 0.0], dtype=np.float32)) == 1.0
    assert cosine(unit, np.array([1.0, 0.0, 0.0], dtype=np.float32)) is None
    assert cosine_scores(unit, [np.array([0.0, 1.0]), np.array([1.0, 0.0, 0.0])]) == [0.0]
