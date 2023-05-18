import click

from tumorsphere.simulation import Simulation

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
def cli(prob_stem, prob_diff, realizations, steps_per_realization, rng_seed):
    """
    Command-line interface for running the tumorsphere simulation.

    Parameters
    ----------
        prob_stem : str
            Comma-separated string of probabilities that a stem cell will self-replicate.
        prob_diff : str
            Comma-separated string of probabilities that a stem cell will yield a differentiated cell.
        realizations : int
            Number of `Culture` objects to simulate for each combination of
            `prob_stem` and `prob_diff`.
        steps_per_realization : int
            Number of steps (tics) per realization.
        rng_seed : int
            Random number generator seed.

    Examples
    --------
    >>> python3 -m tumorsphere.cli --help
    >>> python3 -m tumorsphere.cli --prob-stem "0.5,0.3,0.8" --prob-diff "0.2,0.4,0.6" --realizations 10 --steps-per-realization 10 --rng-seed 1234567
    """
    prob_stem = [float(x) for x in prob_stem.split(',')]
    prob_diff = [float(x) for x in prob_diff.split(',')]

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
        continuous_graph_generation=False,
    )
    sim.simulate_parallel()


if __name__ == "__main__":
    cli()
