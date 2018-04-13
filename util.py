import logging
import sys


def log_setup():
    logger = logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(name)s : %(levelname)s : %(message)s',
                                 level=logging.INFO)
    return logger
