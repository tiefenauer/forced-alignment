# Forced Alignment Demos

This directory contains a lightweight web server that can be used to illustrate some Forced Alignment results from this project. All demos are in the `htdocs` folder. A demo consists of three parts:

- an audio file (`*.wav`)
- an alignment file (`*.json`)
- a HTML file that contains an audio player and a transcription of the audio file

Visualization is done by highlighting the aligned parts while the audio is played. For this the [audiosync](https://github.com/johndyer/audiosync) library was used. The alignment files were created using the VAD-ASR-LSA-pipeline outlined in this project.

To start the web server simply run `python start_server.py` from this directory. You can create your own demos by following the instructions in the [Jupyter Notebook for E2E demonstration](../src/07_e2e.ipynb). Results will be written directly to this folder.

Have fun!