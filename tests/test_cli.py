import os
from pathlib import Path
import filecmp
import shutil

import pandas as pd
import pytest


# ---------- tumorsphere simulate command ----------


@pytest.mark.skip(reason="This test is not working properly yet.")
def test_cli_simulate(run_cli):
    """Test the tumorsphere simulate command with a particular set of
    parameters, to check that it yields the expected outputs.
    """
    cwd = Path(__file__).parent.resolve() / "data/testing_simulate_cli"
    output_dir = cwd / "data"  # This is where the command outputs its files
    expected_dir = cwd / "expected_outputs"

    # Ensure the output directory is clean before running the test
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Run the simulate command
        command = (
            "tumorsphere simulate --prob-stem 0.4,0.8 --prob-diff 0 "
            "--realizations 2 --steps-per-realization 7 --rng-seed 1234 "
            "--sql True --dat-files True --ovito True"
        )
        stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

        assert returncode == 0, (
            f"Simulation command failed with error: {stderr}"
        )

        # Compare the contents of the output and the expected directories
        output_files = sorted(os.listdir(output_dir))
        expected_files = sorted(os.listdir(expected_dir))

        # Check if both directories have the same set of files
        assert output_files == expected_files, (
            "Output and expected directories "
            "do not have the same set of files."
        )
        
        # Compare the content of each file
        for file_name in output_files:
            assert filecmp.cmp(
                output_dir / file_name,
                expected_dir / file_name,
                shallow=False,
            ), f"File contents do not match: {file_name}"
    finally:
        # Cleanup: remove the output directory after the test
        shutil.rmtree(output_dir)


# ---------- tumorsphere status command ----------


def test_cli_status__with_db_files(run_cli):
    """Test the tumorsphere status command with a particular .db file,
    to check that it yields the expected output.
    """
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


def test_cli_status__with_dat_files(run_cli):
    """Test the tumorsphere status command with a particular .dat file,
    to check that it yields the expected output.
    """
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


def test_cli_mergedbs(run_cli):
    """We test the tumorsphere mergedbs command with a particular set of .db
    files, and check that the merged .db file is equal to the correct one.
    """
    # Set the working directory relative to the test file
    cwd = Path(__file__).parent.resolve()

    # Paths for the test
    dbs_folder = cwd / "data/testing_mergedbs_cli"
    merging_path = cwd / "merged.db"
    correct_merged_db = cwd / "data/testing_mergedbs_cli/expected_result/merged.db"

    # Merge the databases
    merge_command = f"tumorsphere mergedbs --dbs-folder {dbs_folder} --merging-path {merging_path}"
    stdout, stderr, returncode = run_cli(merge_command, cwd=str(cwd))
    assert returncode == 0, "Merging databases failed"

    # Compare the newly merged database with the correct version
    compare_command = f"tumorsphere are-dbs-equal --db1 {correct_merged_db} --db2 {merging_path}"
    stdout, stderr, returncode = run_cli(compare_command, cwd=str(cwd))
    assert returncode == 0, "Command failed with an error"
    assert stdout == "Databases are equal.", "The merged databases are not equal"

    # Remove the merged database file
    if merging_path.exists():
        merging_path.unlink()


# ---------- tumorsphere makedf command ----------


def test_cli_makedf__from_db(run_cli):
    """Test the tumorsphere makedf command with a particular .db file,
    to check that it yields the expected output.
    """
    cwd = Path(__file__).parent.resolve()
    db_path = cwd / "data/testing_makedf_cli/merged.db"
    csv_path = cwd / "merged.csv"
    correct_csv_path = cwd / (
        "data/testing_makedf_cli/expected_results/merged.csv"
    )

    # Generate the CSV from the DB file
    command = f"tumorsphere makedf --db-path {db_path} --csv-path {csv_path}"
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))
    assert returncode == 0, "Command failed with an error"

    # Compare the generated CSV with the correct CSV
    generated_df = pd.read_csv(csv_path)
    correct_df = pd.read_csv(correct_csv_path)
    pd.testing.assert_frame_equal(generated_df, correct_df, check_dtype=False)

    # Remove the generated CSV file
    csv_path.unlink()

def test_cli_makedf__from_dat(run_cli):
    """Test the tumorsphere makedf command with a particular .dat file,
    to check that it yields the expected output.
    """
    cwd = Path(__file__).parent.resolve()
    data_dir = cwd / "data/testing_makedf_cli"
    csv_path = cwd / "dat_culture.csv"
    correct_csv_path = cwd / (
        "data/testing_makedf_cli/expected_results/dat_culture.csv"
    )

    # Generate the CSV from the .dat files
    command = f"tumorsphere makedf --db-path {data_dir} --csv-path {csv_path} --dat-files True"
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))
    assert returncode == 0, "Command failed with an error"

    # Compare the generated CSV with the correct CSV
    generated_df = pd.read_csv(csv_path)
    correct_df = pd.read_csv(correct_csv_path)
    pd.testing.assert_frame_equal(generated_df, correct_df, check_dtype=False)

    # Remove the generated CSV file
    csv_path.unlink()


# ---------- tumorsphere are-dbs-equal command ----------


def test_cli_are_dbs_equal__case_databases_are_equal(run_cli):
    """Test the tumorsphere are-dbs-equal command with two identical databases,
    to check that it yields the output: Databases are equal.
    """
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
    """Test the tumorsphere are-dbs-equal command with two different databases,
    to check that it yields the output: Data in table Cells is different.

    I modified the z coordinate of the last cell's position in the second
    database, which was originally a copy of the first one, so the databases
    are different.
    """
    # parent gives me the directory in which the file is located, in this case
    # the directory tests
    cwd = Path(__file__).parent.resolve()
    # we compare the `.db` files from the `tests/data` directory
    command = (
        "tumorsphere are-dbs-equal "
        "--db1 data/testing_are-dbs-equal_cli/merged.db "
        "--db2 data/testing_are-dbs-equal_cli/merged_modified_"
        "_last_cell_position_z_from_-8.312424324276_to_20.0.db"
    )
    stdout, stderr, returncode = run_cli(command, cwd=str(cwd))

    assert returncode == 0, "Command failed with an error"
    assert (
        "Data in table Cells is different." in stdout
    ), "Expected output indicating databases are different was not found"
