"""Script that look at what point in the simulation each culture is."""

import glob
import os
import sqlite3


def count_time_steps_of_dbs_in_dir(data_dir: str) -> None:
    """Count the number of time steps of each culture in the `data_dir`
    directory, and print the steps to the console, indicating the parameters
    for each culture.
    """
    # Iterate through each .db file (with a single culture each)
    for temp_db in glob.glob(os.path.join(data_dir, "*.db")):
        # Create a new database connection for the culture's database
        with sqlite3.connect(temp_db) as conn_temp:
            cursor_temp = conn_temp.cursor()

            # Get the culture's parameters
            cursor_temp.execute(
                "SELECT prob_diff, prob_stem, culture_seed FROM Cultures"
            )
            # Get the result
            pd, ps, seed = cursor_temp.fetchall()[0]
            # fetchall returns a list of tuples, you can also use fetchone
            # instead

            # Get the number of simulation time steps performed (tics)
            cursor_temp.execute("SELECT max(t_creation) FROM Cells")
            # Get the result
            step = cursor_temp.fetchone()[0]

            # We output the status
            print(f"Step {step} for culture pd={pd}, ps={ps}, seed={seed}")


if __name__ == "__main__":
    data_dir = "/home/nate/Devel/tumorsphere_culture/examples/playground/data/"

    count_time_steps_of_dbs_in_dir(data_dir)
