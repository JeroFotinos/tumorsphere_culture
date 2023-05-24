#!/bin/bash


python3 -m tumorsphere.cli --prob-stem "0.6,0.7,0.8" --prob-diff "0" --realizations 5 --steps-per-realization 10 --rng-seed 1234 --parallel-processes 3