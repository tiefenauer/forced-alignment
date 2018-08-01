#!/bin/bash

for poc_id in {1..12}
do
    python ./src/train_poc.py --poc $poc_id
done