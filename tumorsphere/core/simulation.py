"""
Module containing the Simulation class.

Classes:
    - Simulation: Class that manages cultures of cells with different
    parameter combinations, for a given number of realizations per said
    combination.
"""

import multiprocessing as mp
from typing import Tuple, List

import numpy as np

from tumorsphere.core.culture import Culture
from tumorsphere.core.output import create_output_demux
from tumorsphere.core.forces import Force


class Simulation:
    """Class for simulating multiple `Culture` objects with different
    parameters.

    Parameters
    ----------
    first_cell_is_stem : bool, optional
        Whether the first cell of each `Culture` object should be a stem cell.
        If set to `False`, the first cell of the cultures will be a
        differentiated one. Default is `True` (because tumorspheres are CSC
        seeded cultures).
    prob_stem : list of floats, optional
        The probability that a stem cell will self-replicate. Defaults to 0.36
        for being the value measured by Benítez et al. (BMC Cancer, (2021),
        1-11, 21(1))for the experiment of Wang et al. (Oncology Letters,
        (2016), 1355-1360, 12(2)) on a hard substrate.
    prob_diff : list of floats, optional
        The probability that a stem cell will yield a differentiated cell and
        then lose its stemness, effectively yielding two differentiated cells.
        Defaults to 0 (because our intention was to see if percolation occurs,
        and if it doesn't happen at prob_diff = 0, it will never happen).
    num_of_realizations : int, optional
        Number of `Culture` objects to simulate for each combination of
        `prob_stem` and `prob_diff`. Default is `4`.
    num_of_steps_per_realization : int, optional
        Number of simulation steps (i.e., time steps) to perform for each
        `Culture` object. Default is `10`.
    rng_seed : int, optional
        Seed for the random number generator used in the simulation. This is
        the seed on which every other seed depends. Default is the hexadecimal
        number (representing a 128-bit integer)
        `0x87351080E25CB0FAD77A44A3BE03B491`.
    cell_radius : int, optional
        Radius of the cells in the simulation. Default is `1`.
    adjacency_threshold : int, optional
        Distance threshold for two cells to be considered neighbors. Default
        is `4`, which is an upper bound to the second neighbor distance of
        approximately `2 * sqrt(2)` in a hexagonal close packing.
    cell_max_repro_attempts : int, optional
        Maximum number of attempts to create a new cell during the
        reproduction of an existing cell in a `Culture` object.
        Default is`1000`.
    number_of_cells : int, optional
        The number of cells in the culture.
    side : float, optional
        The length of the side of the square where the cells move.
    density : float
        The density of the cells in the culture.
    reproduction : bool
        Whether the cells reproduces or not
    movement : bool
        Whether the cells moves or not

    Attributes
    ----------
    (All parameters, plus the following.)
    rng : `numpy.random.Generator`
        The random number generator used in the simulation to instatiate the
        generator of cultures and cells.
    cultures : dict
        Dictionary storing the `Culture` objects simulated by the `Simulation`.
        The keys are strings representing the combinations of `prob_stem` and
        `prob_diff` and the realization number.


    Methods:
    --------
    simulate_parallel()
        Runs the simulation persisting data to one file for each culture.
    """

    def __init__(
        self,
        force: Force,
        first_cell_is_stem=True,
        prob_stem=[0.36],
        prob_diff=[0],
        num_of_realizations=4,
        num_of_steps_per_realization=10,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius=1,
        adjacency_threshold=4,
        cell_max_repro_attempts=1000,
        swap_probability=0.5,
        number_of_cells=[5],
        density=[0.5],
        reproduction: bool = False,
        movement: bool = True,
    ):
        # main simulation attributes
        self.force = force
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self.swap_probability = swap_probability
        self.number_of_cells = number_of_cells
        self.reproduction = reproduction
        self.movement = movement
        self.density = density
        self._rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # dictionary storing the culture objects
        self.cultures = {}

        # attributes to pass to the culture (and to the cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius

    def simulate_parallel(
        self,
        sql: bool = True,
        dat_files: bool = False,
        dat_pos_ar: bool = False,
        ovito: bool = False,
        number_of_processes: int = None,
        output_dir: str = ".",
    ) -> None:
        """Simulate culture growth `self.num_of_realizations` number of times
        for each combination of self-replication (elements of the
        `self.prob_stem` list) and differentiation probabilities (elements of
        the `self.prob_diff` list), persisting the data of each culture to its
        own file. The simulations are parallelized using multiprocessing.

        Several different output types are simultaneously available, and the
        data that is recorded is handled by the `TumorsphereOutput` classes.
        If `number_of_processes` is None (default), the number of processes is
        equal to the number of cores in the machine. Limitting the number of
        processes is useful when running the simulation in a cluster, where
        the number of cores is limited, or when running with all the resources
        might trigger an alarm.

        Parameters
        ----------
        number_of_processes : int
            The number of the processes. If None (default), the number of
            processes is equal to the number of cores in the machine.
        """
        if number_of_processes is None:
            number_of_processes = mp.cpu_count()

        # Generate seeds for all realizations
        seeds = self.rng.integers(
            low=2**20, high=2**50, size=self.num_of_realizations
        )

        outputs = []
        if sql:
            outputs.append("sql")
        if dat_files:
            outputs.append("dat")
        if dat_pos_ar:
            outputs.append("dat_pos_ar")
        if ovito:
            outputs.append("ovito")

        with mp.Pool(number_of_processes) as p:
            p.map(
                simulate_single_culture,
                [
                    (
                        k,
                        i,
                        f,
                        g,
                        seeds[j],
                        self,
                        outputs,
                        self.force,
                        output_dir,
                    )
                    for k in range(len(self.prob_diff))
                    for i in range(len(self.prob_stem))
                    for f in range(len(self.number_of_cells))
                    for g in range(len(self.density))
                    for j in range(self.num_of_realizations)
                ],
            )


def realization_name(pd, ps, nc, rho, seed, repro, moving) -> str:
    if repro == True and moving == True:
        return f"culture_pd={pd}_ps={ps}_nc={nc}_rho={rho}_rng_seed={seed}"
    elif repro == True and moving == False:
        return f"culture_pd={pd}_ps={ps}_rng_seed={seed}"
    elif repro == False and moving == True:
        return f"culture_nc={nc}_rho={rho}_rng_seed={seed}"
    else:
        pass


def simulate_single_culture(
    args: Tuple[int, int, int, int, int, Simulation, List[str], Force, str]
) -> None:
    """A worker function for multiprocessing.

    This function is used by the multiprocessing.Pool instance in the
    simulate_parallel method to parallelize the simulation of different
    cultures. This simulates the growth of a single culture with the given
    parameters and persists the data.

    Parameters
    ----------
    args : tuple
        A tuple containing the indices for the self-replication probability,
        differentiation probability, the seed to be used in the random number
        generator of the culture, an instance of the Simulation class, and a
        list of strings specifying the desired output types.

    Notes
    -----
    Due to the way multiprocessing works in Python, you can't directly use
    instance methods as workers for multiprocessing. The multiprocessing
    module needs to be able to pickle the target function, and instance
    methods can't be pickled. Therefore, the instance method worker had to be
    refactored to a standalone function (or a static method).
    """
    k, i, f, g, seed, sim, outputs, force, output_dir = args

    current_realization_name = realization_name(
        sim.prob_diff[k],
        sim.prob_stem[i],
        sim.number_of_cells[f],
        sim.density[g],
        seed,
        sim.reproduction,
        sim.movement,
    )
    output = create_output_demux(current_realization_name, outputs, output_dir)
    sim.cultures[current_realization_name] = Culture(
        output,
        force,
        adjacency_threshold=sim.adjacency_threshold,
        cell_radius=sim.cell_radius,
        cell_max_repro_attempts=sim.cell_max_repro_attempts,
        first_cell_is_stem=sim.first_cell_is_stem,
        prob_stem=sim.prob_stem[i],
        prob_diff=sim.prob_diff[k],
        rng_seed=seed,
        swap_probability=sim.swap_probability,
        number_of_cells=sim.number_of_cells[f],
        density=sim.density[g],
        reproduction=sim.reproduction,
        movement=sim.movement,
    )
    sim.cultures[current_realization_name].simulate(
        sim.num_of_steps_per_realization,
    )
