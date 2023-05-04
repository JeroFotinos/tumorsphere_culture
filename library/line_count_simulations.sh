#!/bin/bash

# Commands to see the progress status of the simulations in the cluster.

# This script is intended to use by coping and pasting
# the desired line in the terminal, not by running it directly.

# -------------------------------------------------

# If you want the list of line numbers (time steps)
# of each dat file in a folder, you can do:
wc -l ./*.dat

# If you want the list of booleans that tell you
# if the number of lines of a file it's equal or
# greater than some number, you can do:
for file in ./*.dat; do [ $(wc -l < $file) -ge 60 ] && echo "True" || echo "False"; done

# If instead of the list of values (one for each file)
# you want the count for the number of True and False,
# you can do:
count_true=0; count_false=0; for file in ./*.dat; do if [ $(wc -l < $file) -ge 60 ]; then ((count_true++)); else ((count_false++)); fi; done; echo "True: $count_true, False: $count_false"
