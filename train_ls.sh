#!/bin/bash

declare -a features=("mfcc" "mel" "pow")

for feature_type in "${features[@]}"
do
    echo "training on ${features} features of English speech segments"
    python ./src/train_brnn.py -c rl -l $lang -f $feature_type
done