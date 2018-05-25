import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))
