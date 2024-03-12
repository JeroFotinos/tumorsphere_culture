import click
from tumorsphere.core.simulation import Simulation
from tumorsphere.library.time_step_counter import (
    count_time_steps_of_cultures_in_dir,
)
from tumorsphere.library.dataframe_generation import (
    generate_dataframe_from_db,
    generate_dataframe_from_dat,
)
from tumorsphere.library.db_merger import merge_single_culture_dbs
from tumorsphere.library.db_file_comparer import compare_databases


@click.group()
def cli():
    pass


@click.command(
    help="Simulates tumorsphere growth with the indicated parameters."
)
@click.option(
    "--prob-stem",
    required=True,
    type=str,
    help="List of probabilities for stem cells. "
    "Values should be comma separated.",
)
@click.option(
    "--prob-diff",
    required=True,
    type=str,
    help="List of probabilities for differentiated cells. "
    "Values should be comma separated.",
)
@click.option(
    "--realizations",
    required=True,
    type=int,
    help="Number of realizations.",
)
@click.option(
    "--steps-per-realization",
    required=True,
    type=int,
    help="Number of steps per realization.",
)
@click.option(
    "--rng-seed",
    required=True,
    type=int,
    help="Random number generator seed.",
)
@click.option(
    "--parallel-processes",
    required=False,
    type=int,
    default=None,
    show_default=True,
    help="Number of simultaneous processes. Default is None, which uses all available cores.",
)
@click.option(
    "--sql",
    required=False,
    type=bool,
    default=True,
    show_default=True,
    help="If True, it stores all simulation data in an SQL database.",
)
@click.option(
    "--dat-files",
    required=False,
    type=bool,
    default=False,
    show_default=True,
    help="If True, it only outputs population numbers in a `.dat` file instead of the standard `.db` file.",
)
@click.option(
    "--ovito",
    required=False,
    type=bool,
    default=False,
    show_default=True,
    help="If True, it generates the data for plotting with Ovito instead of the usual data of the simulaiton.",
)
def simulate(
    prob_stem,
    prob_diff,
    realizations,
    steps_per_realization,
    rng_seed,
    parallel_processes,
    sql,
    dat_files,
    ovito,
):
    """
    Command-line interface for running the tumorsphere simulation.

    Parameters
    ----------
        prob_stem : str
            Comma-separated string of probabilities that a stem cell will
            self-replicate.
        prob_diff : str
            Comma-separated string of probabilities that a stem cell will
            yield two differentiated cells.
        realizations : int
            Number of `Culture` objects to simulate for each combination of
            `prob_stem` and `prob_diff`.
        steps_per_realization : int
            Number of steps (tics) per realization.
        rng_seed : int
            Random number generator seed.
        parallel_processes : int, optional
            Number of simultaneous processes. If None (default), uses all
            available cores. When running in a cluster, it should match the
            number of cores requested to the queuing system.
        sql : bool, optional
            True by default. If False, it does not store all simulation data
            in an SQL database.
        dat_files : bool, optional
            False by default. If True, it outputs population numbers in a
            `.dat` file.
        ovito : bool, optional
            False by default. If True, it generates the data for plotting with
            Ovito.


    Examples
    --------
    >>> tumorsphere simulate --help
    >>> tumorsphere simulate --prob-stem "0.6,0.7,0.8" --prob-diff "0" --realizations 5 --steps-per-realization 10 --rng-seed 1234 --parallel-processes 4 --sql False --dat-files True --ovito True
    """
    prob_stem = [float(x) for x in prob_stem.split(",")]
    prob_diff = [float(x) for x in prob_diff.split(",")]

    sim = Simulation(
        first_cell_is_stem=True,
        prob_stem=prob_stem,
        prob_diff=prob_diff,
        num_of_realizations=realizations,
        num_of_steps_per_realization=steps_per_realization,
        rng_seed=rng_seed,
        cell_radius=1,
        adjacency_threshold=4,
        cell_max_repro_attempts=1000,
        # swap_probability=0.5,
    )
    sim.simulate_parallel(
        sql=sql,
        dat_files=dat_files,
        ovito=ovito,
        number_of_processes=parallel_processes,
    )


cli.add_command(simulate)


@click.command(help="Returns the latest simulated time step for each culture.")
@click.option(
    "--data-dir", required=True, type=str, help="Path to the data directory"
)
@click.option(
    "--dat-files",
    required=False,
    type=bool,
    default=False,
    show_default=True,
    help="Use True for a directory with `.dat` files instead of the standard `.db` files.",
)
def status(data_dir, dat_files):
    """Command-line interface that prints the simulation time step each
    tumorsphere culture is in, indicating its parameters for identification
    purposes.

    Parameters
    ----------
        data_dir : str
            Path to the data directory, containing the `.db` files.

    Examples
    --------
    >>> tumorsphere status --help
    >>> tumorsphere status --data-dir ./data
    """
    count_time_steps_of_cultures_in_dir(data_dir, dat_files)


cli.add_command(status)


@click.command(
    help="Merges single culture data bases into a single data base."
)
@click.option(
    "--dbs-folder",
    required=True,
    type=str,
    help="Path to the directory containing the `.db` files to merge",
)
@click.option(
    "--merging-path",
    required=True,
    type=str,
    help="Path and name of the merged `.db` file to append to or create",
)
def mergedbs(dbs_folder, merging_path):
    """Command-line interface that merges single culture data bases into a
    single data base. If the database does not exist, it creates it. If it
    exists, it appends the new data to it.

    Parameters
    ----------
        dbs_folder : str
            Path to the data directory, containing the (unmerged) `.db` files.
        merging_path : str
            Path and name of the `.db` file to append the others to.

    Examples
    --------
    >>> tumorsphere mergedbs --help
    >>> tumorsphere mergedbs --dbs-folder ./data --merging-path ./merged.db
    """
    merge_single_culture_dbs(dbs_folder, merging_path)


cli.add_command(mergedbs)


@click.command(
    help="Makes the DataFrame of population numbers from data base or set of dat files."
)
@click.option(
    "--db-path",
    required=True,
    type=str,
    help="Path and name of the data base to read",
)
@click.option(
    "--csv-path",
    required=True,
    type=str,
    help="Path and name of the `.csv` file to save",
)
@click.option(
    "--dat-files",
    required=False,
    type=bool,
    default=False,
    show_default=True,
    help="If True, it generates the CSV from the set of `.dat` files in --db-path.",
)
def makedf(db_path, csv_path, dat_files):
    """Command-line interface that makes the DataFrame of population numbers
    for each simulated culture in a merged data base.

    Parameters
    ----------
        db_path : str
            Path to the data directory, containing the (merged) `.db` file.
        csv_path : str
            Path and name of the `.csv` file to be generated.

    Examples
    --------
    >>> tumorsphere makedf --help
    >>> tumorsphere makedf --db-path ./merged.db --csv-path ./population_numbers.csv
    """
    if not dat_files:
        generate_dataframe_from_db(db_path, csv_path)
    else:
        df = generate_dataframe_from_dat(db_path, csv_path)
        df.to_csv(csv_path, index=False)


cli.add_command(makedf)


@click.command(help="Compares two databases to check if they are equal.")
@click.option(
    "--db1",
    required=True,
    type=str,
    help="Path and name of the first data base to compare",
)
@click.option(
    "--db2",
    required=True,
    type=str,
    help="Path and name of the second data base to compare",
)
def are_dbs_equal(db1, db2):
    """Compares two databases to check if they are equal.

    Parameters
    ----------
        db1 : str
            Path to the first data base.
        db2 : str
            Path to the second data base.

    Returns
    -------
        bool
            True if the databases are equal, False otherwise.
    """

    return compare_databases(db1, db2)


cli.add_command(are_dbs_equal)


if __name__ == "__main__":
    cli()
