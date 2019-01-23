Pair distances among Fourier series
====================================

Distances between Fourier songs can be calculated indicated
the frequencies and the amplitude, the available metrics can
be found in `distance.py` module.

Distance between two songs is calculated with:

.. autofunction:: foucluster.distance.pair_distance

For a distance of all the songs from a folder, in JSON format
(just the output from `foucluster.transform.transform_folder`,


.. autofunction:: foucluster.distance.distance_matrix
