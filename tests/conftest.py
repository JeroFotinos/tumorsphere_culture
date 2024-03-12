import os

import pytest
import sqlite3
import tempfile
import subprocess

from tumorsphere.core.culture import Culture


@pytest.fixture()
def csc_seeded_culture():
    time_to_simulate = 1
    culture_name_temp = "csc_seeded_culture_fixture"

    try:
        dcc_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=True,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        dcc_culture.simulate(
            num_times=time_to_simulate, culture_name=culture_name_temp
        )

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
    return dcc_culture


@pytest.fixture()
def dcc_seeded_culture():
    time_to_simulate = 1
    culture_name_temp = "dcc_seeded_culture_fixture"

    try:
        dcc_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=False,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        dcc_culture.simulate(
            num_times=time_to_simulate, culture_name=culture_name_temp
        )

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
    return dcc_culture


@pytest.fixture()
def culture_evolved_at_time_2():
    """At time 2 it has been only one timestep since the culture was seeded
    (time starts in 1)."""
    time_to_simulate = 2
    culture_name_temp = "culture_evolved_at_time_2"

    try:
        csc_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=True,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        csc_culture.simulate(
            num_times=time_to_simulate, culture_name=culture_name_temp
        )

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
    return csc_culture


@pytest.fixture()
def culture_evolved_at_time_3():
    """At time 3 it has been two timestep since the culture was seeded (time
    starts in 1)."""
    time_to_simulate = 3
    culture_name_temp = "culture_evolved_at_time_3"
    try:
        csc_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=True,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        csc_culture.simulate(
            num_times=time_to_simulate, culture_name=culture_name_temp
        )

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
    return csc_culture


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
