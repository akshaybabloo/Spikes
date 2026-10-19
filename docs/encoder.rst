Encoders
========

Spike encoders for Spiking Neural Network (spatio-temporal data)

There are three types of encoders:

- :ref:`Bens Spiker Algorithm (BSA) <bsa>`
- :ref:`Threshold Based Representation (TBR) <tbr>`
- :ref:`Step Forward (SF) <sf>`

Step Forward runs faster when `Numba <https://numba.pydata.org/>`_ is installed
(``pip install pyspikes[numba]``). The spikes are the same either way.

.. _bsa:

Bens Spiker Algorithm
---------------------

.. autoclass:: spikes.encoder.BSA
    :members:
    :inherited-members:

.. _tbr:

Threshold Based Representation
------------------------------

.. autoclass:: spikes.encoder.TBR
    :members:
    :inherited-members:

.. _sf:

Step Forward
------------

.. autoclass:: spikes.encoder.StepForward
    :members:
    :inherited-members:
