"""Perform an example simulation and generate data in the appropriate format
for Ovito, for every culture.
"""

from tumorsphere.cells import *
from tumorsphere.culture import *
from tumorsphere.simulation import *


steps_per_realization = 18

prob_diff = [0.0]
prob_stem = [0.7]
realizations = 3
parallel_processes = 3
rng_seed=0x87351080E25CB0FAD77A44A3BE03B491

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
    )

sim.simulate_parallel(number_of_processes=parallel_processes, ovito=True)