"""
Spike Encoders
==============

In this package we will look at some in-built spike encoders.

There are three encoders provided with this library which  can be used as it is or by spiNNaker.

1. Temporal Based Encoding
2. Bens Spiker Algorithm
3. Step Forward
"""

from .bsa import BSA
from .step_forward import StepForward
from .tbr import TBR

__all__ = ["BSA", "TBR", "StepForward"]
