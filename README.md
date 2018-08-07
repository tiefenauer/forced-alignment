# Forced alignment based on speech pauses using an RNN
This repository contains the code for my IP8 project at [FHNW](http://www.fhnw.ch).

## Getting started
The code was written for [Python](https://www.python.org/) `3.6.6` using [Anaconda](https://anaconda.org/) `4.5.8`. The project documentation is available as IPython Notebooks at [https://ip8.tiefenauer.info](https://ip8.tiefenauer.info).
The notebooks provide an interactive way to illustrate the documentation with given or own examples. To run the notebooks on your local machine you must perform some setup work.

### Dependencies
* This application uses [Pydub](http://pydub.com/) which means you will need **libav** or **ffmpeg** on your `PATH`. See the [Pydub Repository](https://github.com/jiaaro/pydub#installation) for further instructions.
* Visual C++ build tools to work with webrtcvad (google it for download link): must be installed before installing the python requirements (see below)!

### Installation
1. Clone [the repository](https://github.com/tiefenauer/forced-alignment): `git clone git@github.com:tiefenauer/forced-alignment.git` 
2. Install Python requirements: `pip install -r requirements.txt`
3. Install [TensorFlow](https://www.tensorflow.org/install/): TF is not included in `requirements.txt` because you can choose between the `tensorflow` (no GPU acceleration) and `tensorflow-gpu` (with GPU-acceleration). If your computer does not have a CUDA-supported GPU (like mine does) you will install the former, else the latter. Installing `tensorflow-gpu` on a computer without GPU does not work (at least I did not get it to work).
3. Run Jupyter Notebook from the directory where you cloned the code into: `jupyter notebook`

### Jupyter Notebook extensions

The following extensions for Jupyter Notebook were used:

* [jupyter_contrib_nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions): A collection of useful extensions (like a TOC) that also includes a manager that allows enabling/disabling individual extensiosn from the web interface
* [cite2c](https://github.com/takluyver/cite2c): For managing citations  (works with [Zotero](https://www.zotero.org/))

## Project anatomy

Most of the code and the documentation is contained in the `src` folder.

| Folder | Description |
|---|---|
| / | root folder containing some Bash scripts to train the RNN |
| assets | binary data (images, audio, etc...) used for the Jupyter Notebooks |
| demos | HTML-files to visualize the result of the alignment pipeline. |
| src | scripts and Python source files containing all application logic. Also, the documentation is stored here |
| test | some unit tests |
| tmp | temporary folder, e.g. needed for the VAD stage. No persistent files should be stored here as this folder might be deleted at any time by application logic! |