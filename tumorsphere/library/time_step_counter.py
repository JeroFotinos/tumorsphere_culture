"""Script that look at what point in the simulation each culture is."""

import glob
import os
import sqlite3


def count_time_steps_of_cultures_in_dir(
    data_dir: str, dat_files: bool = False
) -> None:
    """
    Print the time steps of each culture in the `data_dir` directory.

    Count the number of time steps of each culture in the `data_dir`
    directory, and print the steps to the console, indicating the parameters
    for each culture.

    Parameters
    ----------
    data_dir : str
        The directory path where the .dat or .db files are located.
    dat_files : bool, optional
        If True, the function will count lines in .dat files. If False, it
        will query .db files. Default is False.

    Returns
    -------
    None
        This function prints to the console the number of time steps for each
        file. Note that this has to be in accordance with the tumorsphere
        simulate command, so time starts at 0 (with a single cell), and the
        final time (number of steps) is `--steps-per-realization`.
    """
    if dat_files:
        # Handle .dat files
        for dat_file in glob.glob(os.path.join(data_dir, "*.dat")):
            with open(dat_file, "r") as f:
                line_count = sum(1 for line in f)
                print(f"{os.path.basename(dat_file)}: {line_count-2} steps.")
                # we subtract 2 because the first line is the header, and the
                # second line is the initial state of the culture (time 0)
                # so the number of steps (final time, since we start at 0) is
                # the number of lines minus 2
    else:
        # Handle .db files
        for temp_db in glob.glob(os.path.join(data_dir, "*.db")):
            with sqlite3.connect(temp_db) as conn_temp:
                cursor_temp = conn_temp.cursor()

                # Get the culture's parameters
                cursor_temp.execute(
                    "SELECT prob_diff, prob_stem, culture_seed FROM Cultures"
                )
                pd, ps, seed = cursor_temp.fetchall()[0]

                # Get the number of simulation time steps performed (tics)
                cursor_temp.execute("SELECT max(t_creation) FROM Cells")
                step = cursor_temp.fetchone()[0]

                print(
                    f"{step} steps for culture pd={pd}, ps={ps}, seed={seed}"
                )
                # creation time starts at 0 in the database, so no
                # modification is needed


if __name__ == "__main__":
    data_dir = "/home/nate/Devel/tumorsphere_culture/examples/playground/data/"

    count_time_steps_of_cultures_in_dir(data_dir)
