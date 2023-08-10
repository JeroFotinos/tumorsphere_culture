#!/bin/bash

# This formats the ending of the file, indicating the timestep,
# as a two digit index. No longer needed.

# Assuming you're in the directory with the files
for file in ovito_data_*.*; do
    # Extracts the number after the last dot in the filename
    number=$(echo $file | rev | cut -d. -f1 | rev)
    
    # Check if number is a single digit
    if [ ${#number} -eq 1 ]; then
        base_name=$(echo $file | rev | cut -d. -f2- | rev)
        new_file="${base_name}.0${number}"
        
        mv "$file" "$new_file"
    fi
done
