"""
This module contains functions for loading data from the simulations and saving it to a
pandas.DataFrame, which can in turn be saved to a CSV file. Functions in this module
can be imported without further consequences, but direct execution of this file will
trigger the file processing in the last block of code.

Example
-------
To use this module, simply run it from the command line:

    $ python3 dataframe_generation.py
"""

import glob
import pandas as pd
import os


def extract_params_from_filename(filename):
    """
    Extract the parameters pd, ps and n from the filename.

    The parameters that characterize a simulation are `ps` (self-replication probability
    of the CSCs), `pd` (probability that a CSC will yield a DCC) and the realization
    number `n`. The names of the files are supposed to be on the format
    `culture_pd=<pd>_ps=<ps>_realization_<n>`, e.g. `culture_pd=0.2_ps=0.3_realization_0`.

    Parameters
    ----------
    filename : str
        The filename from which to extract the parameters.

    Returns
    -------
    pd, ps, n : tuple of float, float, int
        The extracted parameters.

    Notes
    -----
    Parameter extraction from the filename is abstracted and separated into its own
    function to keep the main data loading function cleaner and more focused. This can be
    particularly useful if the file name parsing logic becomes more complex in the future.

    """
    # starting in 1 to remove the string "culture"
    # "_" is a placeholder for the string "realization"
    p_d, p_s, _, n = filename.replace(".dat", "").split("_")[1:]
    p_d = p_d.replace("pd=", "")
    p_s = p_s.replace("ps=", "")
    n = n.replace("realization", "")

    return float(p_d), float(p_s), int(n)


def load_simulation_data(data_dir):
    """
    Load simulation data from a directory of .dat files into a pandas DataFrame.

    Reads the files in the indicated directory. The files are supposed to be CSVs,
    containing four columns: numbers of total cells, active cells, stem cells and
    active stem cells; in that order, as a function of time (each row representing
    a time step). Also, the first row is skipped cause it contains headers.

    Parameters
    ----------
    data_dir : str
        The path to the directory where the data files are stored.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the combined data from all simulations.

    Notes
    -----
    Parameter extraction form file name is delegated to the
    extract_params_from_filename function.

    Examples
    --------
    >>> df = load_simulation_data("/path/to/data/dir")
    >>> print(df.head())
    """
    # Define the columns of the final dataframe
    cols = [
        "pd",
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
    for file_path in glob.glob(os.path.join(data_dir, "*.dat")):
        # Extract the values of pd, ps, and n from the file name
        try:
            p_d, p_s, n = extract_params_from_filename(
                os.path.basename(file_path)
            )
        except ValueError:
            print(f"Skipping file {file_path} due to improper format.")
            continue

        # Load the data from the file into a temporary dataframe
        try:
            temp_df = pd.read_csv(
                file_path,
                header=None,
                names=[
                    "total_cells",
                    "active_cells",
                    "stem_cells",
                    "active_stem_cells",
                ],
                skiprows=1,
            )
        except pd.errors.ParserError:
            print(f"Skipping file {file_path} due to parsing error.")
            continue

        # Add columns for pd, ps, n, and time to the temporary dataframe
        temp_df["pd"] = p_d
        temp_df["ps"] = p_s
        temp_df["n"] = n
        temp_df["time"] = range(1, len(temp_df) + 1)

        # Reorder the columns of the temporary dataframe
        temp_df = temp_df[cols]

        # Append the temporary dataframe to the final dataframe
        df = pd.concat([df, temp_df], ignore_index=True)

    # Convert the columns to the desired data types
    df[["n", "time"]] = df[["n", "time"]].astype(int)
    df[["ps", "pd"]] = df[["ps", "pd"]].astype(float)

    return df


if __name__ == "__main__":
    data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_7_short_time_broad_ps/dat_files/"
    save_path = "/home/nate/Devel/tumorsphere_culture/data/sim_7_short_time_broad_ps/post-processing/df_simulations.csv"

    df = load_simulation_data(data_dir)
    print(df.head())
    df.to_csv(save_path, index=False)
