#!/bin/sh

### Script to submit the simulation.
### You can execute it by doing:
### qsub ./submit_simulation_measuring_time.sh

# Record the start time
start_time=$(date +%s)

# Submit the simulation command
# python3 -m tumorsphere.cli --prob-stem 0.5 0.3 0.8 --prob-diff 0.2 0.4 0.6 --realizations 10 --steps-per-realization 10 --rng-seed 1234567
# python3 -m tumorsphere.cli --prob-stem 0.5 --prob-stem 0.3 --prob-stem 0.8 --prob-diff 0.2 --prob-diff 0.4 --prob-diff 0.6 --realizations 10 --steps-per-realization 100 --rng-seed 1234567
python3 -m tumorsphere.cli --prob-stem "0.5,0.3,0.8" --prob-diff "0.2,0.4,0.6" --realizations 10 --steps-per-realization 10 --rng-seed 1234567

# Record the end time
end_time=$(date +%s)

# Calculate the execution time
execution_time=$((end_time - start_time))

# Display the execution time
echo "Execution Time: $execution_time seconds"
