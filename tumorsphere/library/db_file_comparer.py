"""Compares to `.db` files to see if they are EXACTLY the same.

Might not be useful if different ordering of rows doesn't matter.
"""

import sqlite3


def compare_databases(db_path1, db_path2):
    """Compare two SQLite databases.

    Parameters
    ----------
    db_path1 : str
        The path to the first database.
    db_path2 : str
        The path to the second database.
    """
    # Connect to both databases
    conn1 = sqlite3.connect(db_path1)
    conn2 = sqlite3.connect(db_path2)

    # Get the schema for both databases
    schema1 = conn1.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    schema2 = conn2.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    # Compare the schema
    if schema1 != schema2:
        print("Schemas are different.")
        return False

    # Iterate through each table and compare data
    for table_name, _ in schema1:
        rows1 = conn1.execute(
            f"SELECT * FROM {table_name} ORDER BY rowid"
        ).fetchall()
        rows2 = conn2.execute(
            f"SELECT * FROM {table_name} ORDER BY rowid"
        ).fetchall()

        if rows1 != rows2:
            print(f"Data in table {table_name} is different.")
            # we print the differences
            for i, (row1, row2) in enumerate(zip(rows1, rows2)):
                if row1 != row2:
                    print(
                        (
                            f"Row {i} is different:\n"
                            f"\tFirst DB: {row1}\n"
                            f"\tSecond DB: {row2}"
                        )
                    )
            return False

    print("Databases are equal.")
    return True


if __name__ == "__main__":
    db_path1 = (
        "/home/nate/Devel/tumorsphere_culture/examples/"
        "playground/merged_new_into_existing.db"
    )
    db_path2 = (
        "/home/nate/Devel/tumorsphere_culture/examples/"
        "playground/merged_from_individual_cultures.db"
    )
    compare_databases(db_path1, db_path2)
