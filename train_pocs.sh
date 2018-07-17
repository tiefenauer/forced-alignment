#!/bin/bash

for poc_id in {8..12}
do
    python ./src/train_rnn.py --poc $poc_id
done