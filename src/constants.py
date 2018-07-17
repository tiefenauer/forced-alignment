import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))

CORPUS_RAW_ROOT = r'D:\corpus' if os.name == 'nt' else '/media/all/D1/'  # root directory for raw corpus files
CORPUS_TARGET_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default corpus directory
TRAIN_TARGET_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default directory for training results