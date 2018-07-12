#!/bin/bash

for poc_id in {1..12}
do
    python ./src/train_rnn.py --poc $poc_id
done