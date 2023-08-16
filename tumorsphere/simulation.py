"""
Module containing the Simulation class.

Classes:
    - Simulation: Class that manages cultures of cells with different
    parameter combinations, for a given number of realizations per said
    combination.
"""
import multiprocessing as mp
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from tumorsphere.culture import Culture


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
    ):
        # main simulation attributes
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self.swap_probability = swap_probability
        self._rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
    
        # dictionary storing the culture objects
        self.cultures = {}

        # attributes to pass to the culture (and cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius

    def simulate_parallel(
        self, ovito: bool = False, number_of_processes: int = None
    ) -> None:
        """ Simulate culture growth `self.num_of_realizations` number of times
        for each combination of self-replication (elements of the
        `self.prob_stem` list) and differentiation probabilities (elements of
        the `self.prob_diff` list), realizations and persists the data of each
        culture to its own file. The simulations are parallelized using
        multiprocessing.

        The data of the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells, is
        persisted to a file with a name specifying the parameters in the
        format `f'culture_pd={pd}_ps={ps}_rng_seed={culture_seed}.dat'`. You
        can specify the number of simultaneous processes to use in the
        simulation with the `number_of_processes` parameter. If
        `number_of_processes` is None (default), the number of processes is
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
        seeds = self.rng.integers(low=2**20, high=2**50, size=self.num_of_realizations)
        
        if ovito:
            with mp.Pool(number_of_processes) as p:
                p.map(
                    simulate_single_culture_ovito,
                    [
                        (k, i, seeds[j], self)
                        for k in range(len(self.prob_diff))
                        for i in range(len(self.prob_stem))
                        for j in range(self.num_of_realizations)
                    ],
                )
        else:
            with mp.Pool(number_of_processes) as p:
                p.map(
                    simulate_single_culture,
                    [
                        (k, i, seeds[j], self)
                        for k in range(len(self.prob_diff))
                        for i in range(len(self.prob_stem))
                        for j in range(self.num_of_realizations)
                    ],
                )


def simulate_single_culture(
    args: Tuple[int, int, int, Simulation]
) -> None:
    """A worker function for multiprocessing.

    This function is used by the multiprocessing.Pool instance in the
    simulate_parallel method to parallelize the simulation of different
    cultures. This simulates the growth of a single culture with the given
    parameters and persist the data.

    Parameters
    ----------
    args : tuple
        A tuple containing the indices for the self-replication probability,
        differentiation probability, the seed to be used in the random number
        generator of the culture, and the instance of the Simulation class.

    Notes
    -----
    Due to the way multiprocessing works in Python, you can't directly use
    instance methods as workers for multiprocessing. The multiprocessing
    module needs to be able to pickle the target function, and instance
    methods can't be pickled. Therefore, the instance method worker had to be
    refactored to a standalone function (or a static method).
    """
    k, i, seed, sim = args
    current_realization_name = (
        f"culture_pd={sim.prob_diff[k]}_ps={sim.prob_stem[i]}_rng_seed={seed}"
    )
    sim.cultures[current_realization_name] = Culture(
        adjacency_threshold=sim.adjacency_threshold,
        cell_radius=sim.cell_radius,
        cell_max_repro_attempts=sim.cell_max_repro_attempts,
        first_cell_is_stem=sim.first_cell_is_stem,
        prob_stem=sim.prob_stem[i],
        prob_diff=sim.prob_diff[k],
        rng_seed=seed,
        swap_probability = sim.swap_probability,
    )
    sim.cultures[current_realization_name].simulate_with_persistent_data(
        sim.num_of_steps_per_realization,
        current_realization_name,
    )


def simulate_single_culture_ovito(
    args: Tuple[int, int, int, Simulation]
) -> None:
    """Copy of simulate_single_culture for Ovito plotting. A worker function
    for multiprocessing.

    This function is used by the multiprocessing.Pool instance in the
    simulate_parallel method to parallelize the simulation of different
    cultures. This simulates the growth of a single culture with the given
    parameters and persist the data.

    Parameters
    ----------
    args : tuple
        A tuple containing the indices for the self-replication probability,
        differentiation probability, the seed to be used in the random number
        generator of the culture, and the instance of the Simulation class.

    Notes
    -----
    Due to the way multiprocessing works in Python, you can't directly use
    instance methods as workers for multiprocessing. The multiprocessing
    module needs to be able to pickle the target function, and instance
    methods can't be pickled. Therefore, the instance method worker had to be
    refactored to a standalone function (or a static method).
    """
    k, i, seed, sim = args
    current_realization_name = (
        f"culture_pd={sim.prob_diff[k]}_ps={sim.prob_stem[i]}_rng_seed={seed}"
    )
    sim.cultures[current_realization_name] = Culture(
        adjacency_threshold=sim.adjacency_threshold,
        cell_radius=sim.cell_radius,
        cell_max_repro_attempts=sim.cell_max_repro_attempts,
        first_cell_is_stem=sim.first_cell_is_stem,
        prob_stem=sim.prob_stem[i],
        prob_diff=sim.prob_diff[k],
        rng_seed=seed
    )
    sim.cultures[current_realization_name].simulate_with_ovito_data(
        sim.num_of_steps_per_realization,
        current_realization_name,
    )
