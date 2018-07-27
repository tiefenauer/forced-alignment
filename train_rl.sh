#!/bin/bash

declare -a languages=("de" "fr" "it" "en")
declare -a features=("mfcc" "mel" "pow")

for lang in "${languages[@]}"
do
    for feature_type in "${features[@]}"
    do
        echo "training on ${features} features of speech segments with language=${lang}"
        python ./src/train_brnn.py -c rl -l $lang -f $feature_type
    done
done