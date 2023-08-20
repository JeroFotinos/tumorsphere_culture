"""This module provides functions to merge temporary single-culture SQLite
databases into a single consolidated database. The schema involves tables for
representing cultures (of cells), cells (individually), and stemness changes.

### Direct Execution of the Module

Direct execution will add all data contained in `.db` files in `data_dir` to
the existing `save_path` database. If the `save_path` database does not exist,
yet, it creates it.


### Resulting Database Schema Details

1. **Cultures Table**:
   - `culture_id`: primary key, auto-incrementing.
   - `prob_stem`: probability of CSC self-replication, real number, not null.
   - `prob_diff`: probability that a CSC yields two DCCs after reproduction,
    real number, not null.
   - `culture_seed`: the culture's rng seed, integer, not null.
   - `simulation_start`: timestamp indicating the simulation start time, not
    null.
   - `adjacency_threshold`: adjacency threshold, real number, not null.
   - `swap_probability`: probability that a CSC interchanges position with its
    DCC daughter, real number, not null.

2. **Cells Table**:
   - `cell_id`: primary key, auto-incrementing.
   - `_index`: original index from temporary database, integer, not null.
   - `parent_index`: index of the parent cell, nullable integer.
   - `position_x`, `position_y`, `position_z`: real numbers for spatial
    coordinates, not null.
   - `t_creation`: simulation time (tic) of creation, integer, not null.
   - `t_deactivation`: simulation time (tic) of deactivation, nullable
    integer.
   - `culture_id`: foreign key referencing the `Cultures` table.

3. **StemChanges Table**:
   - `change_id`: primary key, auto-incrementing.
   - `cell_id`: foreign key referencing the `Cells` table, integer, not null.
   - `t_change`: simulation time (tic) of the change, integer, not null.
   - `is_stem`: boolean flag indicating if the cell is a stem cell, not null.


#### Temporary Database vs. Resulting Database

While both schemas are simmilar, there are some key differences:
- **Cultures Table**: for temporary databases, created during the simulation
    of a culture, there is only one record corresponding to that culture. On
    the contrary, the resulting database contains as many culure records as
    temporary databases merged.
- **Cells Table**: `_index` is the primary key for temporary databases, but
    after merging, a new `cell_id` key is used to maintain a consistent
    reference across the merged data. However, the original `_index` values
    are kept to retain the meaning of the `parent` field.
- **StemChanges Table**: while in temporary databases `cell_id` references
    `_index`, in the merged database it references `cell_id`.

Note: the module relies on standard libraries like `glob`, `os`, and `sqlite3`
for file handling and database connectivity.
"""

import glob
import os
import sqlite3
from sqlite3 import Connection


def create_tables(conn: Connection) -> None:
    """Create tables for Cultures, Cells, and StemChanges.

    This function creates the required tables if they do not exist in the
    provided SQLite3 database connection.

    Parameters
    ----------
    conn : sqlite3.Connection
        SQLite3 connection object to the database.

    Notes
    -----
    The tables created are:
    - Cultures: Contains culture-related attributes.
    - Cells: Contains individual cell attributes and references to Cultures.
    - StemChanges: Contains information on changes in stem cell state,
      referencing Cells.
    """
    with conn:
        cursor = conn.cursor()

        # Enable foreign key constraints for this connection
        cursor.execute("PRAGMA foreign_keys = ON;")

        # Creating the Culture table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Cultures (
                culture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prob_stem REAL NOT NULL,
                prob_diff REAL NOT NULL,
                culture_seed INTEGER NOT NULL,
                simulation_start TIMESTAMP NOT NULL,
                adjacency_threshold REAL NOT NULL,
                swap_probability REAL NOT NULL
            );
        """
        )

        # Creating the Cells table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS Cells (
            cell_id INTEGER PRIMARY KEY AUTOINCREMENT,
            _index INTEGER NOT NULL,
            parent_index INTEGER,
            position_x REAL NOT NULL,
            position_y REAL NOT NULL,
            position_z REAL NOT NULL,
            t_creation INTEGER NOT NULL,
            t_deactivation INTEGER,
            culture_id INTEGER,
            FOREIGN KEY(culture_id) REFERENCES Cultures(culture_id)
        );
        """
        )

        # Creating the StemChanges table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS StemChanges (
            change_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cell_id INTEGER NOT NULL,
            t_change INTEGER NOT NULL,
            is_stem BOOLEAN NOT NULL,
            FOREIGN KEY(cell_id) REFERENCES Cells(cell_id)
        );
        """
        )


def merge_db_files_in_dir_into_conn(
    conn_merged: Connection, data_dir: str
) -> None:
    """
    Merge data from temporary SQLite databases into a consolidated database.

    This function takes all temporary .db files from the specified directory
    and merges them into a given consolidated database. It ensures that data
    like Cells and StemChanges are correctly referenced in the new database.

    Parameters
    ----------
    conn_merged : sqlite3.Connection
        SQLite3 connection object to the merged database.
    data_dir : str
        Path to the directory containing the temporary .db files.

    Notes
    -----
    The function manages the migration of data from temporary databases to the
    merged one. It takes care of mapping the original `_index` to new
    `cell_id` in the merged database.
    """

    # Context manager for the connection
    with conn_merged:
        # Create a new database connection for the merged database
        cursor_merged = conn_merged.cursor()

        # Iterate through the temporary .db files
        for temp_db in glob.glob(os.path.join(data_dir, "*.db")):
            # Create a new database connection for the temporary database
            with sqlite3.connect(temp_db) as conn_temp:
                cursor_temp = conn_temp.cursor()

                # Copy Culture
                cursor_temp.execute(
                    "SELECT * FROM Cultures"
                )  # this should be a single row
                cultures = cursor_temp.fetchall()
                assert len(cultures) == 1
                for culture in cultures:
                    cursor_merged.execute(
                        """
                        INSERT INTO Cultures (prob_stem, prob_diff, culture_seed, simulation_start, adjacency_threshold, swap_probability)
                        VALUES (?, ?, ?, ?, ?, ?);
                    """,
                        (
                            float(culture[1]),
                            float(culture[2]),
                            int(culture[3]),
                            culture[4],
                            int(culture[5]),
                            float(culture[6]),
                        ),
                    )
                    culture_id = cursor_merged.lastrowid

                # Keep track of mapping from _index to new cell_id
                _index_to_cell_id = {}

                # Copy Cells
                cursor_temp.execute("SELECT * FROM Cells")
                cells = cursor_temp.fetchall()
                for cell in cells:
                    # Insert into merged database and get the new cell_id
                    cursor_merged.execute(
                        """
                        INSERT INTO Cells (_index, parent_index, position_x, position_y, position_z, t_creation, culture_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                        (
                            int(cell[0]),
                            int(cell[1]),
                            float(cell[2]),
                            float(cell[3]),
                            float(cell[4]),
                            int(cell[5]),
                            culture_id,
                        ),
                    )
                    _index_to_cell_id[int(cell[0])] = cursor_merged.lastrowid

                # Copy StemChange
                cursor_temp.execute("SELECT * FROM StemChanges")
                stem_changes = cursor_temp.fetchall()
                for change in stem_changes:
                    # Get the new cell_id using the mapping
                    cell_id = _index_to_cell_id[
                        int(change[1])
                    ]  # change[0] is the change_id
                    cursor_merged.execute(
                        """
                        INSERT INTO StemChanges (cell_id, t_change, is_stem)
                        VALUES (?, ?, ?);
                    """,
                        (
                            cell_id,
                            int(change[2]),
                            bool(change[3]),
                        ),
                    )


if __name__ == "__main__":
    data_dir = (
        "/home/nate/Devel/tumorsphere_culture/examples/playground/data/"
    )
    save_path = (
        "/home/nate/Devel/tumorsphere_culture/examples/playground/merged.db"
    )

    with sqlite3.connect(save_path) as conn:
        create_tables(conn)
        merge_db_files_in_dir_into_conn(conn, data_dir)
