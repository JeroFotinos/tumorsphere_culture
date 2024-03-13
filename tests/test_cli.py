from pathlib import Path


# ---------- tumorsphere simulate command ----------


# ---------- tumorsphere status command ----------


def test_status_cli_with_db_files(run_cli):
    cwd = Path(__file__).parent.resolve()
    data_dir = "data/testing_status_cli"

    command = f"tumorsphere status --data-dir {data_dir}"
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    expected_output = (
        "10 steps for culture pd=0.0, ps=0.3, seed=322927469740366"
    )
    assert (
        expected_output in stdout
    ), (
        "Expected output indicating steps for .db files was not found. "
        f"Found: {stdout}"
    )


def test_status_cli_with_dat_files(run_cli):
    cwd = Path(__file__).parent.resolve()
    data_dir = "data/testing_status_cli"

    command = f"tumorsphere status --data-dir {data_dir} --dat-files True"
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    expected_output = (
        "culture_pd=0.0_ps=0.3_rng_seed=322927469740366.dat: 10 steps."
    )
    assert (
        expected_output in stdout
    ), (
        "Expected output indicating steps for .dat files was not found. "
        f"Found: {stdout}"
    )


# ---------- tumorsphere mergedbs command ----------


# ---------- tumorsphere makedf command ----------


# ---------- tumorsphere are-dbs-equal command ----------


def test_cli_are_dbs_equal__case_databases_are_equal(run_cli):
    # parent gives me the directory in which the file is located, in this case
    # the directory tests
    cwd = Path(__file__).parent.resolve()
    # we compare the `.db` files from the `tests/data` directory
    command = (
        "tumorsphere are-dbs-equal "
        "--db1 data/testing_are-dbs-equal_cli/merged.db "
        "--db2 data/testing_are-dbs-equal_cli/merged_identical_copy.db"
    )
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    assert (
        stdout == "Databases are equal."
    ), "Expected output indicating databases are equal was not found"


def test_cli_are_dbs_equal__case_databases_are_different(run_cli):
    # parent gives me the directory in which the file is located, in this case
    # the directory tests
    cwd = Path(__file__).parent.resolve()
    # we compare the `.db` files from the `tests/data` directory
    command = (
        "tumorsphere are-dbs-equal "
        "--db1 data/testing_are-dbs-equal_cli/merged.db "
        "--db2 data/testing_are-dbs-equal_cli/merged_modified_"
        "_last_cell_position_z_from_1.74092095578382_to_20.0.db"
    )
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    assert (
        "Data in table Cells is different." in stdout
    ), "Expected output indicating databases are different was not found"
