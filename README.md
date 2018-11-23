# FouCluster

[![Build Status](https://travis-ci.org/cperales/foucluster.svg?branch=master)](https://travis-ci.org/cperales/foucluster)
[![Coverage Status](https://coveralls.io/repos/github/cperales/foucluster/badge.svg?branch=master)](https://coveralls.io/github/cperales/foucluster?branch=master)

*This project was presented at [PyCon ES 2018](https://2018.es.pycon.org/).
PDF presentation can be found in spanish
[here](https://es.slideshare.net/CarlosPerales/clustering-de-canciones-usando-fourier)
Video of the presentation in spanish can be found in [Youtube](https://www.youtube.com/watch?v=a9AJSfIEbo0)*

## Installation and use

Requirements are already added in `setup.py`, so you just need to run

```bash
python setup.py install
``` 

A [documented example](https://cperales.github.io/foucluster/) is available.
Firstly, creative commons songs for the example must be downloaded
from this [Dropbox link](https://www.dropbox.com/s/sidecqpk6dhgsdq/song.zip?dl=0).
Unzip the songs, so `.mp3` files can be found in `song/` folder.

After that, just run

```bash
python test/test_full.py
```

Clustering with other songs can be done by adding them into `song/` folder.


## Motivation
Recommendation song systems nowadays, like **Spotify**, use song clustering by made up
[parameters](https://www.theverge.com/tldr/2018/2/5/16974194/spotify-recommendation-algorithm-playlist-hack-nelson)
such as *danceability*, *energy*, *instrumentalness*, ... etc, which need an expert in that area to create those
parameters.

In order to avoid expert knowledge and make access to machine
learning applied to song easier, this library
use signal analysis for measuring distances between songs.
With this distances, when the amount of songs is considerable clustering
can be applied.

Because [musical notes have associated frequencies](https://www.intmath.com/trigonometric-graphs/music.php),
this proposal is based on transforming from time series to frequency series, and then grouping theses series
using various techniques and distance metrics.
