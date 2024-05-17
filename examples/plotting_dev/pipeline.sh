#!/bin/bash

# Simulate and save data to sim_output_data
# tumorsphere simulate --prob-stem "0.6,0.7" --prob-diff "0.0,0.1" --realizations 2 --steps-per-realization 12 --rng-seed 23 --sql True --dat-files True --output-dir ./sim_output_data

# Merge dbs from sim_output_data to merged_df_and_db
tumorsphere mergedbs --dbs-folder ./sim_output_data/ --merging-path ./merged_df_and_db/merged.db

# Get DataFrame from merged db
tumorsphere makedf --db-path ./merged_df_and_db/merged.db --csv-path ./merged_df_and_db/df_from_dbs.csv

# Get DataFrame from dat files
tumorsphere makedf --db-path ./sim_output_data/ --csv-path ./merged_df_and_db/df_from_dats.csv --dat-files True
