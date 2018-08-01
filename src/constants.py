import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))

CORPUS_RAW_ROOT = r'D:\corpus' if os.name == 'nt' else '/media/all/D1/'  # root directory for raw corpus files
CORPUS_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default corpus directory
RL_CORPUS_ROOT = join(CORPUS_ROOT, 'readylingua-corpus')
LS_CORPUS_ROOT = join(CORPUS_ROOT, 'librispeech-corpus')

DEMO_ROOT = join(ROOT_DIR, 'demos', 'htdocs')

# PoC profiles
# @formatter:off
POC_PROFILES = {
    'poc_1': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'mfcc',
        'limit_segments': 5
    },
    'poc_2': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'mfcc',
        'limit_segments': 5, 'synthesize': True
    },
    'poc_3': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'mel',
        'limit_segments': 5
    },
    'poc_4': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'mel',
        'limit_segments': 5, 'synthesize': True
    },
    'poc_5': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'pow',
        'limit_segments': 5
    },
    'poc_6': {
        'corpus': 'rl', 'language': 'de', 'id': 'andiefreudehokohnerauschenrein', 'feature_type': 'pow',
        'limit_segments': 5, 'synthesize': True
    },
    'poc_7': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'mfcc', 'limit_segments': 5
    },
    'poc_8': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'mfcc', 'limit_segments': 5, 'synthesize': True
    },
    'poc_9': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'mel', 'limit_segments': 5
    },
    'poc_10': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'mel', 'limit_segments': 5, 'synthesize': True
    },
    'poc_11': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'pow', 'limit_segments': 5
    },
    'poc_12': {
        'corpus': 'rl', 'language': 'en', 'id': 'sunday22ohnerauschen', 'feature_type': 'pow', 'limit_segments': 5, 'synthesize': True
    }
}
# // @formatter:on

# selected BCP-47 codes
LANGUAGE_CODES = {
    'en': 'en-US',
    'de': 'de-DE',
    'it': 'it-IT',
    'fr': 'fr-FR',
    'es': 'es-ES'
}

# default parameters for training
CORPUS = 'ls'
LANGUAGE = 'en'
NUM_EPOCHS = 20
NUM_STEPS_TRAIN = 0
NUM_STEPS_VAL = 0
BATCH_SIZE = 5
FEATURE_TYPE = 'mfcc'
TRAIN_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default directory for training results
ARCHITECTURE = 'ds1'