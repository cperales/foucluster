Example with plotting
=======================

When you have installed `mpg123` or `ffmpeg` for MP3 to WAV
transform, you should download
`this zip <https://www.dropbox.com/s/sidecqpk6dhgsdq/song.zip>`_
and unzip it. Then, run

.. code-block:: bash

    python test/test_full

This test:

- Load the configuration from ``config.ini``.
- Transform the songs from `song/` to a temporary folder.
- Transform each song from WAV to a Fourier series, plot and store it.
- Calculate the distances between songs.
- Cluster them.

Examples of the Fourier series:

*Higher and Higher* from *Scream Inc* is a rock song.

.. image:: image/Higher_And_Higher_rock.png


*Secret mission* from *Frank Rawel* is a jazz song.

.. image:: image/Secret_Mission_jazz.png

The distance matrix can be plot as a heat map,
where the dark values means the songs are closer and
white means the distances are high.

.. image:: image/integrate.png

If you can read the heat map, you already can see there
are two groups. These two groups are jazz music and
rock music. So we can apply cluster algorithms to the
system.

.. image:: image/score.png
