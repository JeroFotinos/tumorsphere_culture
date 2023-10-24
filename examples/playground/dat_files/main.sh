#!/bin/bash

# Initialize j to indicate the index of the generated files
j=1

# Loop through the range of prob-stem values
for i in $(seq 0.65 0.01 0.76); do
  # Increment the next value by 0.01 to generate the pair of prob-stem values
  next_val=$(echo "$i + 0.01" | bc)

  # Create the file name
  file_name="submit_sim_13_${j}.sh"

  # Copy the template and make modifications
  sed "s/--job-name=tumorsphere_13_1/--job-name=tumorsphere_13_${j}/g;
       s/--prob-stem \"0.65,0.66\"/--prob-stem \"${i},${next_val}\"/g" \
    submit_sim_13.sh > $file_name

  # Make the generated file executable
  chmod +x $file_name

  # Increment j for the next iteration
  ((j++))
done

# Loop to submit jobs to the queue
j=1
for i in $(seq 0.65 0.01 0.76); do
  # sbatch "submit_sim_13_${j}.sh"
  echo "sbatch submit_sim_13_${j}.sh"
  ((j++))
done