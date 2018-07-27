#!/bin/bash

for lang in ['de', 'fr', 'it', 'en']
do
    python ./src/train_brnn.py -c rl -l $lang
done