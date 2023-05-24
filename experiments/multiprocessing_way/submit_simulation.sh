#!/bin/sh

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation.sh

python3 -m tumorsphere.cli --prob-stem "0.5,0.3,0.8" --prob-diff "0.2,0.4,0.6" --realizations 10 --steps-per-realization 10 --rng-seed 1292317634567