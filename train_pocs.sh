#!/bin/bash

for poc_id in {9..12}
do
    python ./src/train_poc.py --poc $poc_id
done