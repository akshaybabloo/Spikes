"""The NumPy and Numba backends must give exactly the same spikes."""

import numpy as np
import numpy.testing as npt
import pytest

from spikes import _kernels

pytestmark = pytest.mark.skipif(not _kernels.HAS_NUMBA, reason="Numba is not installed")

rng = np.random.default_rng(7)
DATA = np.cumsum(rng.normal(size=(9, 300, 5)), axis=1)  # Random walks: smooth enough to spike in both directions


def test_step_forward_backends_match():
    threshold = rng.uniform(0.1, 2, size=(DATA.shape[0], DATA.shape[2]))

    expected = _kernels.step_forward_numpy(DATA, threshold)

    npt.assert_array_equal(_kernels.step_forward_numba(DATA, threshold), expected)
