import os

import pytest
import sqlite3
import tempfile
import subprocess


# -------- For testing the `db_file_comparer.py` module --------


@pytest.fixture
def temp_db():
    """Fixture to create a temporary database file with specific content."""

    def _temp_db(schema, data):
        temp_db = tempfile.NamedTemporaryFile(delete=False)
        conn = sqlite3.connect(temp_db.name)
        cursor = conn.cursor()
        for table, create_statement in schema.items():
            cursor.execute(create_statement)
            for row in data.get(table, []):
                cursor.execute(
                    f"INSERT INTO {table} VALUES ({','.join('?' for _ in row)})",
                    row,
                )
        conn.commit()
        conn.close()
        return temp_db.name

    return _temp_db


# -------- For testing `tumorsphere tumorsphere are-dbs-equal` --------
# --------    (the CLI of the `db_file_comparer.py` module)    --------


@pytest.fixture(scope="session")
def run_cli():
    """Fixture to run a CLI command and return its output, error, and exit status."""

    def _run_cli(command, cwd=None):
        result = subprocess.run(
            command, capture_output=True, text=True, shell=True, cwd=cwd
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode

    return _run_cli
