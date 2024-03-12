from pathlib import Path


# ---------- tumorsphere simulate command ----------


# ---------- tumorsphere status command ----------


# ---------- tumorsphere mergedbs command ----------


# ---------- tumorsphere makedf command ----------


# ---------- tumorsphere are-dbs-equal command ----------


def test_cli_are_dbs_equal__case_databases_are_equal(run_cli):
    # parent gives me the directory in which the file is located, in this case
    # the directory tests
    cwd = Path(__file__).parent.resolve()
    # we compare the `.db` files from the `tests/data` directory
    command = (
        "tumorsphere are-dbs-equal --db1 data/merged.db --db2 "
        "data/merged_identical_copy.db"
    )
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    assert (
        stdout == "Databases are equal."
    ), "Expected output indicating databases are equal was not found"


def test_cli_databases_are_different(run_cli):
    # parent gives me the directory in which the file is located, in this case
    # the directory tests
    cwd = Path(__file__).parent.resolve()
    # we compare the `.db` files from the `tests/data` directory
    command = (
        "tumorsphere are-dbs-equal --db1 data/merged.db "
        "--db2 data/merged_modified_"
        "_last_cell_position_z_from_1.74092095578382_to_20.0.db"
    )
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    assert (
        "Data in table Cells is different." in stdout
    ), "Expected output indicating databases are different was not found"
