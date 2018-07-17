import os

from os.path import abspath, dirname, join

SRC_DIR = dirname(abspath(__file__))
ROOT_DIR = abspath(join(SRC_DIR, os.pardir))

CORPUS_RAW_ROOT = r'D:\corpus' if os.name == 'nt' else '/media/all/D1/'  # root directory for raw corpus files
CORPUS_TARGET_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default corpus directory
TRAIN_TARGET_ROOT = r'E:\\' if os.name == 'nt' else '/media/all/D1'  # default directory for training results

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
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'mfcc', 'limit_segments': 5
    },
    'poc_8': {
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'mfcc', 'limit_segments': 5, 'synthesize': True
    },
    'poc_9': {
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'mel', 'limit_segments': 5
    },
    'poc_10': {
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'mel', 'limit_segments': 5, 'synthesize': True
    },
    'poc_11': {
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'pow', 'limit_segments': 5
    },
    'poc_12': {
        'corpus': 'ls', 'language': 'en', 'ix': 0, 'feature_type': 'pow', 'limit_segments': 5, 'synthesize': True
    }
}
# // @formatter:on