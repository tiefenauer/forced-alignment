# Forced alignment based on speech pauses using an RNN
This repository contains the code for my IP8 project at [FHNW](http://www.fhnw.ch).

## Getting started
The code was written for [Python](https://www.python.org/) `3.6.6` using [Anaconda](https://anaconda.org/) `4.5.8`.

### Installation
1. Clone this repository: `git clone `
2. Install requirements: `pip install -r requirements.txt`
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work (at least I did not get it to work).
3. Run Jupyter Notebook: `jupyter notebook`

### Dependencies
* This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.
* Visual C++ build tools to work with webrtcvad (google it for download link)

## Project anatomy
The code of this project is structured into the following packages:

| package/subdirectory | description |
|---|---|
| src | contains jupyter notebooks and main python scripts |
| src/corpus | class definitions for corpora |
| src/util | various utility scripts |
| test | test scripts |
| assets | assets for jupyter notebooks |

## Jupyter notebooks
The following jupyter notebooks illustrate the whole project process interactively:

| Jupyter notebook | description |
|---|---|
| [01_create_corpora](./src/01_create_corpora.ipynb) | How corpora were created from raw data |
| [02_create_labelled_data](./src/02_create_labelled_data.ipynb) | How labelled data was created from corpus data |
| [03_train_rnn](./src/03_train_rnn.ipynb) | RNN architecture and training information |


## Jupyter Notebook extensions

The following extensions for Jupyter Notebook were used:

* [jupyter_contrib_nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions): A collection of useful extensions (like a TOC) that also includes a manager that allows enabling/disabling individual extensiosn from the web interface
* [cite2c](https://github.com/takluyver/cite2c): For managing citations made with [Zotero](https://www.zotero.org/)