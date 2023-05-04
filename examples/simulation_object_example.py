from tumorsphere.cells import *
from tumorsphere.culture import *
from tumorsphere.simulation import *

sim = Simulation(
    first_cell_is_stem=True,
    prob_stem=[0.36],  # Wang HARD substrate value
    prob_diff=[0],
    num_of_realizations=3,
    num_of_steps_per_realization=4,
    rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
    cell_radius=1,
    adjacency_threshold=4,
    cell_max_repro_attempts=1000,
    continuous_graph_generation=False,
)

sim.simulate()

fig, ax = sim.plot_average_data(ps_index=0, pd_index=0)
plt.show()
