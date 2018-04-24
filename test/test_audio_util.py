import os
import unittest

import audio_util


class AudioUtilTest(unittest.TestCase):

    def test_mp3_to_wav(self):
        infile = os.path.abspath('208.mp3')
        outfile = os.path.abspath('208.wav')
        audio_util.mp3_to_wav(infile, outfile)