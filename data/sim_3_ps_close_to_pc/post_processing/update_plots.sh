#!/bin/bash

# Save the current directory
pushd .

# Change directory to where the scripts are located
cd /home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/post_processing

# Execute each Python script in order
python3 generate_average_data.py
python3 generate_p_infty_average_data.py
python3 plot_active_stem_average.py
python3 plot_stem_total_average.py
python3 plot_p_infty_vs_t_averages.py
python3 plot_p_infty_vs_ps_averages.py
python3 df_for_simulations.py
python3 simulation_df_exploration.py

# Return to the original directory
popd