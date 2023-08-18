import glob
import os
import sqlite3


def create_tables(conn):
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

# Iterate through the temporary .db files
def merge_db_files_in_dir_into_conn(conn_merged, data_dir):
    with conn_merged:
        # Create a new database connection for the merged database
        cursor_merged = conn_merged.cursor()

        for temp_db in glob.glob(os.path.join(data_dir, "*.db")):
            # Create a new database connection for the temporary database
            with sqlite3.connect(temp_db) as conn_temp:
                cursor_temp = conn_temp.cursor()

                # Copy Culture
                cursor_temp.execute("SELECT * FROM Cultures") # this should be a single row
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
                    cell_id = _index_to_cell_id[int(change[1])] # change[0] is the change_id
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
    data_dir = "/home/nate/Devel/tumorsphere_culture/examples/playground/data/"
    save_path = "/home/nate/Devel/tumorsphere_culture/examples/playground/merged.db"

    with sqlite3.connect(save_path) as conn:
        create_tables(conn)
        merge_db_files_in_dir_into_conn(conn, data_dir)
