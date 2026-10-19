import pytest

from spikes import _kernels

numba_only = pytest.mark.skipif(not _kernels.HAS_NUMBA, reason="Numba is not installed")


@pytest.fixture(params=["numpy", pytest.param("numba", marks=numba_only)])
def backend(request):
    """Name of an encoding backend in :mod:`spikes._kernels`; tests using it run once per backend."""
    return request.param
