"""
This module generate a pandas `DataFrame` with population numbers and data of
the culture they are from, using the `.db` database or the old `.dat` files.
Functions in this module can be imported without further consequences, but
direct execution of this file will trigger the file processing in the last
block of code.

Example
-------
First, make sure to set the `db_files` bool, specifying if the input data is
a `.db` file. Then, to use this module simply run it from the command line:

    $ python3 dataframe_generation.py
"""

import glob
import os
import sqlite3

import pandas as pd

# ------------------------ Functions for `.dat` files ------------------------


def extract_params_from_filename(filename):
    """
    Extract the parameters pd, ps and seed from the name of the `.dat` file.

    The parameters that characterize a simulation are `ps` (self-replication
    probability of the CSCs), `pd` (probability that a CSC will yield a DCC),
    and the seed of the rng, `seed`. The names of the files are supposed to be
    on the format `culture_pd={pd}_ps={ps}_rng_seed={seed}.dat`, e.g.
    `culture_pd=0.0_ps=0.36_rng_seed=12341234.dat`.

    Parameters
    ----------
    filename : str
        The filename from which to extract the parameters.

    Returns
    -------
    pd, ps, seed: tuple of float, float, int
        The extracted parameters.

    Notes
    -----
    Parameter extraction from the filename is abstracted and separated into its own
    function to keep the main data loading function cleaner and more focused. This can be
    particularly useful if the file name parsing logic becomes more complex in the future.

    """
    # starting in 1 to remove the string "culture"
    # "_" is a placeholder for the string "rng"
    p_d, p_s, _, the_seed = filename.replace(".dat", "").split("_")[1:]
    p_d = float(p_d.replace("pd=", ""))
    p_s = float(p_s.replace("ps=", ""))
    the_seed = int(the_seed.replace("seed=", ""))

    return p_d, p_s, the_seed


def generate_dataframe_from_dat(data_dir):
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
    >>> df = generate_dataframe_from_dat("/path/to/data/dir")
    >>> print(df.head())
    """
    # Define the columns of the final dataframe
    cols = [
        "pd",
        "ps",
        "rng_seed",
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
            p_d, p_s, seed = extract_params_from_filename(
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

        # Add columns for pd, ps, rng_seed, and time to the temporary dataframe
        temp_df["pd"] = p_d
        temp_df["ps"] = p_s
        temp_df["rng_seed"] = seed
        temp_df["time"] = range(len(temp_df))

        # Reorder the columns of the temporary dataframe
        temp_df = temp_df[cols]

        # Append the temporary dataframe to the final dataframe
        df = pd.concat([df, temp_df], ignore_index=True)

    # Convert the columns to the desired data types
    df[["rng_seed", "time"]] = df[["rng_seed", "time"]].astype(int)
    df[["ps", "pd"]] = df[["ps", "pd"]].astype(float)

    return df


# ------------------------ Functions for `.db` Data Base ---------------------


def generate_dataframe_from_db(db_path: str, csv_path_and_name: str):
    """Generate a pandas DataFrame from the given SQLite database and save it
    as a CSV file.

    The DataFrame will contain a row for each simulation time per culture,
    with the following columns:
    - "culture_id": the culture_id value for the culture referred to by that row
    - "pd": the value of prob_diff for that culture
    - "ps": the value of prob_stem for that culture
    - "rng_seed": the culture_seed value for that culture
    - "time": the simulation time. For each culture, there will be a row for each integer value starting from 0, up to the greatest t_creation of the cells in that culture.
    - "total_cells": the number of cells in that culture with t_creation less or equal than the 'time' value for that row.
    - "active_cells": the number of cells in that culture with null t_deactivation, or t_deactivation greater than the 'time' value for that row.
    - "stem_cells": the number of cells that are stem cells at that time in that culture.
    - "active_stem_cells": similar to 'stem_cells', but only counting the cells that also have null t_deactivation, or t_deactivation greater than the 'time' value for that row.

    Parameters
    ----------
    db_path : str
        The path to the SQLite database containing the merged simulation data.
    csv_path : str
        The path where the resulting CSV file should be saved.

    Returns
    -------
    None
        The function saves the DataFrame to the specified CSV file and does not return any value.

    Raises
    ------
    sqlite3.OperationalError
        If there is any issue with the database connection or queries.
    IOError
        If there is any issue with saving the CSV file.

    Examples
    --------
    >>> generate_dataframe("merged_database.sqlite", "output_dataframe.csv")
    """

    # Connect to the SQLite database
    with sqlite3.connect(db_path) as conn:
        # Query to get the maximum simulation time per culture
        max_time_query = """
        SELECT culture_id, MAX(t_creation) as max_time
        FROM Cells
        GROUP BY culture_id
        """

        # Define the columns of the final dataframe
        cols = [
            "culture_id",
            "pd",
            "ps",
            "rng_seed",
            "time",
            "total_cells",
            "active_cells",
            "stem_cells",
            "active_stem_cells",
        ]

        # Create a DataFrame to hold the final result
        result_df = pd.DataFrame(columns=cols)
        row_idx = 0  # Initialize a row index

        # Iterate over each culture and its max simulation time
        for culture_id, max_time in conn.execute(max_time_query).fetchall():
            # Query to get the culture details
            culture_query = f"""
            SELECT prob_diff, prob_stem, culture_seed
            FROM Cultures
            WHERE culture_id = {culture_id}
            """
            prob_diff, prob_stem, rng_seed = conn.execute(
                culture_query
            ).fetchone()

            # Iterate over each simulation time for the current culture
            for time in range(max_time + 1):
                # Query to get total_cells and active_cells
                cells_query = f"""
                SELECT COUNT(*),
                    COUNT(CASE WHEN t_deactivation IS NULL OR t_deactivation > {time} THEN 1 END)
                FROM Cells
                WHERE culture_id = {culture_id} AND t_creation <= {time}
                """
                total_cells, active_cells = conn.execute(
                    cells_query
                ).fetchone()

                # Query to get stem_cells
                stem_cells_query = f"""
                SELECT COUNT(*)
                FROM (
                    SELECT cell_id, MAX(t_change) as latest_time, MAX(change_id) as latest_change_id
                    FROM StemChanges
                    WHERE t_change <= {time}
                    GROUP BY cell_id
                ) AS latest_changes
                JOIN StemChanges ON StemChanges.cell_id = latest_changes.cell_id AND StemChanges.change_id = latest_changes.latest_change_id
                JOIN Cells USING (cell_id)
                WHERE is_stem = 1 AND culture_id = {culture_id} AND t_creation <= {time}
                """
                stem_cells = conn.execute(stem_cells_query).fetchone()[0]

                # Query to get active_stem_cells
                active_stem_cells_query = f"""
                SELECT COUNT(*)
                FROM (
                    SELECT cell_id, MAX(t_change) as latest_time, MAX(change_id) as latest_change_id
                    FROM StemChanges
                    WHERE t_change <= {time}
                    GROUP BY cell_id
                ) AS latest_changes
                JOIN StemChanges ON StemChanges.cell_id = latest_changes.cell_id AND StemChanges.change_id = latest_changes.latest_change_id
                JOIN Cells USING (cell_id)
                WHERE is_stem = 1 AND culture_id = {culture_id} AND t_creation <= {time} AND (t_deactivation IS NULL OR t_deactivation > {time})
                """
                active_stem_cells = conn.execute(
                    active_stem_cells_query
                ).fetchone()[0]

                # Create a new row
                new_row = {
                    "culture_id": culture_id,
                    "pd": prob_diff,
                    "ps": prob_stem,
                    "rng_seed": rng_seed,
                    "time": time,
                    "total_cells": total_cells,
                    "active_cells": active_cells,
                    "stem_cells": stem_cells,
                    "active_stem_cells": active_stem_cells,
                }

                # Add the new row to the result DataFrame using loc
                result_df.loc[row_idx] = new_row
                row_idx += 1  # Increment the row index

        # Save the DataFrame as a CSV file
        result_df.to_csv(csv_path_and_name, index=False)


# ------------------------ Instructions for Module Execution -----------------


if __name__ == "__main__":
    # set to `False` if you are trying to process the old `.dat` files, and
    # leave it as `True` for processing the standard `.db` merged database.
    db_files = True

    # directories for `.dat` files
    dat_data_dir = "/home/nate/Devel/tumorsphere_culture/examples/multiprocessing_example/data/"
    dat_save_path = "/home/nate/Devel/tumorsphere_culture/examples/multiprocessing_example/df_simulations.csv"

    # directories for `.db` Data Base
    db_path = (
        "/home/nate/Devel/tumorsphere_culture/examples/playground/merged.db"
    )
    db_csv_path_and_name = "/home/nate/Devel/tumorsphere_culture/examples/playground/df_simulations.csv"

    if db_files:
        generate_dataframe_from_db(db_path, db_csv_path_and_name)
    else:
        df = generate_dataframe_from_dat(dat_data_dir)
        print(df.head())
        df.to_csv(dat_save_path, index=False)
