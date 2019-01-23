Transform audio into fourier series
=====================================

For computing de distances among the songs, this library uses de frequencies
of the analogous Fourier Series from the songs. This frequencies are obtained
with :code:`numpy.fft` function.

Because :code:`numpy.fft` only reads WAV, firstly songs must be transformed
from MP3 to WAV. In this library, we use this function:

.. autofunction:: foucluster.transform.mp3_to_wav

Once MP3 songs are encoded to WAV, Fourier transform is applied. To avoid
too much useless information, frequencies are limited (to 6000 Hz by default)
and grouped by a step size.

.. autofunction:: foucluster.transform.wav_to_fourier

Both process can be executed at one, included a plot of the Fourier series,
with the following function.

.. autofunction:: foucluster.transform.time_to_frequency

When this last function wants to be executed for a whole folder, with or
without multiprocessing, this can be done with the main function

.. autofunction:: foucluster.transform.transform_folder
