import glob

import pandas as pd

# Define the path to the directory where the data files are stored
data_dir = (
    "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/dat_files"
)

# Define the columns of the final dataframe
cols = [
    "ps",
    "n",
    "time",
    "total_cells",
    "active_cells",
    "stem_cells",
    "active_stem_cells",
]

# Create an empty dataframe to hold the data from all the simulations
df = pd.DataFrame(columns=cols)

# Loop over all the data files in the directory
for file_path in glob.glob(data_dir + "/*.dat"):
    # Extract the values of ps and n from the file name
    ps, n = file_path.split("/")[-1].replace(".dat", "").split("-")

    # Load the data from the file into a temporary dataframe
    temp_df = pd.read_csv(
        file_path,
        header=None,
        names=[
            "total_cells",
            "active_cells",
            "stem_cells",
            "active_stem_cells",
        ],
    )

    # Add columns for ps, n, and time to the temporary dataframe
    temp_df["ps"] = float(ps)
    temp_df["n"] = int(n)
    temp_df["time"] = range(1, len(temp_df) + 1)

    # Reorder the columns of the temporary dataframe
    temp_df = temp_df[cols]

    # Append the temporary dataframe to the final dataframe
    df = pd.concat([df, temp_df], ignore_index=True)

# Convert the columns to the desired data types
df[["n", "time"]] = df[["n", "time"]].astype(int)
df["ps"] = df["ps"].astype(float)

# Print the resulting dataframe
print(df.head())

# Save the dataframe as a CSV file
df.to_csv(
    "/home/nate/Devel/tumorsphere_culture/data/sim_3_ps_close_to_pc/p_infty_vs_ps_fit/df_simulations.csv",
    index=False,
)
