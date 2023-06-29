#!/bin/bash

python -m tumorsphere.cli --prob-stem "0.3,0.8" --prob-diff "0" --realizations 1 --steps-per-realization 8 --rng-seed 12345 --parallel-processes 2

