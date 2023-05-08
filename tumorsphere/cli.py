import click

from tumorsphere.simulation import Simulation


@click.command()
@click.option("--prob-stem", required=True, type=float)
def cli(prob_stem):
    sim = Simulation(
        first_cell_is_stem=True,
        prob_stem=[prob_stem],  # Wang HARD substrate value
        num_of_realizations=1,
        num_of_steps_per_realization=60,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius=1,
        adjacency_threshold=4,
        cell_max_repro_attempts=1000,
        continuous_graph_generation=False,
    )
    sim.simulate()


if __name__ == "__main__":
    cli()
