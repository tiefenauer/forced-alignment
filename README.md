# Forced alignment based on speech pauses using an RNN
This repository contains the code for my IP8 project at [FHNW](http://www.fhnw.ch).

## Getting started
The code was written for [Python](https://www.python.org/) `3.6.4` using [Anaconda](https://anaconda.org/) `4.4.11`.

### Installation
1. Clone this repository: `git clone `
2. Install requirements: `pip install -r requirements.txt`
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work (at least I did not get it to work).
3. Run Jupyter Notebook: `jupyter notebook`

This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.

The notebook ip8.ipynb contains some examples on how to create corpora and train the RNN.

