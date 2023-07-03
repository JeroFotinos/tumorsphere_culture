import click

from tumorsphere.simulation import SimulationLite


@click.command(
    help="Command-line interface for running the tumorsphere simulation."
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
def cli(
    prob_stem,
    prob_diff,
    realizations,
    steps_per_realization,
    rng_seed,
    parallel_processes,
):
    """
    Command-line interface for running the tumorsphere simulation.

    Parameters
    ----------
        prob_stem : str
            Comma-separated string of probabilities that a stem cell will self-replicate.
        prob_diff : str
            Comma-separated string of probabilities that a stem cell will yield two differentiated cells.
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
            number of cores requested to the queueing system.

    Examples
    --------
    >>> python3 -m tumorsphere.cli --help
    >>> python3 -m tumorsphere.cli --prob-stem "0.6,0.7,0.8" --prob-diff "0" --realizations 5 --steps-per-realization 10 --rng-seed 1234 --parallel-processes 4
    """
    prob_stem = [float(x) for x in prob_stem.split(",")]
    prob_diff = [float(x) for x in prob_diff.split(",")]

    sim = SimulationLite(
        first_cell_is_stem=True,
        prob_stem=prob_stem,
        prob_diff=prob_diff,
        num_of_realizations=realizations,
        num_of_steps_per_realization=steps_per_realization,
        rng_seed=rng_seed,
        cell_radius=1,
        adjacency_threshold=4,
        cell_max_repro_attempts=1000,
        # continuous_graph_generation=False, # for Simulation
    )
    sim.simulate_parallel(parallel_processes)


if __name__ == "__main__":
    cli()
