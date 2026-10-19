import numpy as np
import pytest

from spikes.encoder.base import HAS_NUMBA

numba_only = pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed")


@pytest.fixture(params=["numpy", pytest.param("numba", marks=numba_only)])
def backend(request):
    """Name of an encoding backend (``numpy`` or ``numba``); tests using it run once per backend."""
    return request.param


@pytest.fixture
def random_walks():
    """Random walks of shape ``(9, 300, 5)``: smooth enough to spike in both directions."""
    return np.cumsum(np.random.default_rng(7).normal(size=(9, 300, 5)), axis=1)
