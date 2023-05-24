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
    """
    Class for simulating multiple `Culture` objects with different parameters.

    Parameters
    ----------
    first_cell_is_stem : bool, optional
        Whether the first cell of each `Culture` object should be a stem cell
        or a differentiated one. Default is `True` (because tumorspheres are
        CSC-seeded cultures).
    prob_stem : list of floats, optional
        The probability that a stem cell will self-replicate. Defaults to 0.36
        for being the value measured by Benítez et al. (BMC Cancer, (2021),
        1-11, 21(1))for the experiment of Wang et al. (Oncology Letters,
        (2016), 1355-1360, 12(2)) on a hard substrate.
    prob_diff : list of floats, optional
        The probability that a stem cell will yield a differentiated cell.
        Defaults to 0 (because the intention was to see if percolation occurs,
        and if it doesn't happen at prob_diff = 0, it will never happen).
    num_of_realizations : int, optional
        Number of `Culture` objects to simulate for each combination of
        `prob_stem` and `prob_diff`. Default is `10`.
    num_of_steps_per_realization : int, optional
        Number of simulation steps to perform for each `Culture` object.
        Default is `10`.
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
    continuous_graph_generation : bool, optional
        Whether to update the adjacency graph after each cell division, or to
        keep it empty until manual generation of the graph. Default is `False`.

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
    data : dict
        Dictionary storing the simulation data for each `Culture` object
        simulated by the `Simulation`. The keys are strings representing
        the combinations of `prob_stem` and `prob_diff` and the realization
        number.
    average_data : dict
        Dictionary storing the average simulation data for each combination of
        `prob_stem` and `prob_diff`. The keys are strings representing the
        combinations of `prob_stem` and `prob_diff`.

    Methods:
    --------
    simulate()
        Runs the simulation persisting data to one file for each culture.
    simulate_parallel()
        Runs the simulation persisting data to one file for each culture.
    simulate_small()
        Runs the simulation saving data to a dictionary for interactive use.
    _average_of_data_ps_i_and_pd_k(i, k)
        Computes the average of the data for a given pair of probabilities
        `prob_stem[i]` and `prob_diff[k]`.
    plot_average_data(ps_index, pd_index)
        Plot the average data for a given combination of self-replication
        and differentiation probabilities.
    """

    def __init__(
        self,
        first_cell_is_stem=True,
        prob_stem=[0.36],  # Wang HARD substrate value
        prob_diff=[0],  # p_d; probability that a CSC gives a DCC and then
        # loses stemness (i.e. prob. that a CSC gives two DCCs)
        num_of_realizations=10,
        num_of_steps_per_realization=10,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius=1,
        adjacency_threshold=4,  # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        cell_max_repro_attempts=1000,
        continuous_graph_generation=False,
        # THE USER MUST PROVIDE A HIGH QUALITY SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # main simulation attributes
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self._rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # dictionary storing the culture objects
        self.cultures = {}

        # dictionary storing simulation data (same keys as the previous one)
        self.data = {}

        # array with the average evolution over realizations, per p_s, per p_d
        self.average_data = {}
        for pd in self.prob_diff:
            for ps in self.prob_stem:
                self.average_data[f"average_pd={pd}_ps={ps}"] = {
                    "total": np.zeros(self.num_of_steps_per_realization),
                    "active": np.zeros(self.num_of_steps_per_realization),
                    "total_stem": np.zeros(self.num_of_steps_per_realization),
                    "active_stem": np.zeros(self.num_of_steps_per_realization),
                }

        # attributes to pass to the culture (and cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.continuous_graph_generation = continuous_graph_generation

    def simulate_small(self):
        """Simulate the culture growth for different self-replication and
        differentiation probabilities and realizations and compute the average
        data for each of the self-replication and differentiation probability
        combinations.

        The data of the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells, is
        saved in a dictionary, both for every single culture and the average
        over the realizations. It's recommended for small simulations because
        it saves the data to a dictionary, without persisting it.
        """
        for k in range(len(self.prob_diff)):
            for i in range(len(self.prob_stem)):
                for j in range(self.num_of_realizations):
                    # we compute a string with the ps and number of this realization
                    current_realization_name = f"culture_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}_realization_{j}"
                    # we instantiate the culture of this realization as an item of
                    # the self.cultures dictionary, with the string as key
                    self.cultures[current_realization_name] = Culture(
                        adjacency_threshold=self.adjacency_threshold,
                        cell_radius=self.cell_radius,
                        cell_max_repro_attempts=self.cell_max_repro_attempts,
                        first_cell_is_stem=self.first_cell_is_stem,
                        prob_stem=self.prob_stem[i],
                        prob_diff=self.prob_diff[k],  # implementar en culture
                        continuous_graph_generation=self.continuous_graph_generation,
                        rng_seed=self.rng.integers(low=2**20, high=2**50),
                    )
                    # we simulate the culture's growth and retrive data in the
                    # self.data dictionary, with the same string as key
                    self.data[current_realization_name] = self.cultures[
                        current_realization_name
                    ].simulate_with_data(self.num_of_steps_per_realization)
                # now we compute the averages
                self.average_data[
                    f"average_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}"
                ] = self._average_of_data_ps_i_and_pd_k(i, k)

    def _average_of_data_ps_i_and_pd_k(self, i, k):
        """Compute the average data for a combination of self-replication and
        differentiation probabilities for all realizations.

        Parameters
        ----------
        i : int
            Index of the self-replication probability.
        k : int
            Index of the differentiation probability.

        Returns
        -------
        average : dict
            A dictionary with the average data for the given combination of
            self-replication and differentiation probabilities.
        """
        # For prob_stem[i] and prob_diff[k], we average
        # over the j realizations (m is a string)
        data_of_ps_i_and_pd_k_realizations = {}
        for j in range(self.num_of_realizations):
            data_of_ps_i_and_pd_k_realizations[j] = self.data[
                f"culture_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}_realization_{j}"
            ]

        # we stack the data for all variables and average it:
        # total
        vstacked_total = data_of_ps_i_and_pd_k_realizations[0]["total"]
        for j in range(1, self.num_of_realizations):
            vstacked_total = np.vstack(
                (
                    vstacked_total,
                    data_of_ps_i_and_pd_k_realizations[j]["total"],
                )
            )
        average_total = np.mean(
            vstacked_total,
            axis=0,
        )

        # active
        vstacked_active = data_of_ps_i_and_pd_k_realizations[0]["active"]
        for j in range(1, self.num_of_realizations):
            vstacked_active = np.vstack(
                (
                    vstacked_active,
                    data_of_ps_i_and_pd_k_realizations[j]["active"],
                )
            )
        average_active = np.mean(
            vstacked_active,
            axis=0,
        )

        # total_stem
        vstacked_total_stem = data_of_ps_i_and_pd_k_realizations[0][
            "total_stem"
        ]
        for j in range(1, self.num_of_realizations):
            vstacked_total_stem = np.vstack(
                (
                    vstacked_total_stem,
                    data_of_ps_i_and_pd_k_realizations[j]["total_stem"],
                )
            )
        average_total_stem = np.mean(
            vstacked_total_stem,
            axis=0,
        )

        # active_stem
        vstacked_active_stem = data_of_ps_i_and_pd_k_realizations[0][
            "active_stem"
        ]
        for j in range(1, self.num_of_realizations):
            vstacked_active_stem = np.vstack(
                (
                    vstacked_active_stem,
                    data_of_ps_i_and_pd_k_realizations[j]["active_stem"],
                )
            )
        average_active_stem = np.mean(
            vstacked_active_stem,
            axis=0,
        )

        # we organaize data in the appropriate format for storing in
        # self.average_data[f"average_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}"]
        average = {
            "total": average_total,
            "active": average_active,
            "total_stem": average_total_stem,
            "active_stem": average_active_stem,
        }
        return average

    def plot_average_data(self, ps_index, pd_index):
        """Plot the average data for a given combination of self-replication
        and differentiation probabilities.

        Parameters
        ----------
        ps_index : int
            Index of the self-replication probability.
        pd_index : int
            Index of the differentiation probability.

        Examples
        --------
        >>> fig, ax = simulation.plot_average_data()
        >>> plt.show()
        """
        # create a figure and axis objects
        fig, ax = plt.subplots()

        # plot each row of the array with custom labels and colors
        data = self.average_data[
            f"average_pd={self.prob_diff[pd_index]}_ps={self.prob_stem[ps_index]}"
        ]

        ax.plot(data["total"], label="Total", color="blue")
        ax.plot(
            data["active"],
            label="Total active",
            color="green",
        )
        ax.plot(data["total_stem"], label="Stem", color="orange")
        ax.plot(data["active_stem"], label="Active stem", color="red")

        # set the title and axis labels
        ax.set_title("Average evolution of culture")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Number of cells")

        # create a legend and display the plot
        ax.legend()

        return fig, ax

    def simulate(self):
        """Simulate the culture growth for different self-replication and
        differentiation probabilities and realizations and persists the data
        of each culture to its own file.

        The data of the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells, is
        persisted to a file with a name specifying the parameters in the
        format culture_pd={prob_diff}_ps={prob_stem}_realization_{j}.dat.
        """
        for k in range(len(self.prob_diff)):
            for i in range(len(self.prob_stem)):
                for j in range(self.num_of_realizations):
                    # we compute a string with the ps and number of this realization
                    current_realization_name = f"culture_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}_realization_{j}"
                    # we instantiate the culture of this realization as an item of
                    # the self.cultures dictionary, with the string as key
                    self.cultures[current_realization_name] = Culture(
                        adjacency_threshold=self.adjacency_threshold,
                        cell_radius=self.cell_radius,
                        cell_max_repro_attempts=self.cell_max_repro_attempts,
                        first_cell_is_stem=self.first_cell_is_stem,
                        prob_stem=self.prob_stem[i],
                        prob_diff=self.prob_diff[k],
                        continuous_graph_generation=self.continuous_graph_generation,
                        rng_seed=self.rng.integers(low=2**20, high=2**50),
                    )
                    # we simulate the culture's growth and retrive data in the
                    # self.data dictionary, with the same string as key
                    self.cultures[
                        current_realization_name
                    ].simulate_with_persistent_data(
                        self.num_of_steps_per_realization,
                        current_realization_name,
                    )

    def simulate_parallel(self, number_of_processes: int = None) -> None:
        """
        Simulate the culture growth for different self-replication and
        differentiation probabilities and realizations and persists the data
        of each culture to its own file. The simulations are parallelized
        using multiprocessing.

        The data of the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells, is
        persisted to a file with a name specifying the parameters in the
        format culture_pd={prob_diff}_ps={prob_stem}_realization_{j}.dat. You
        can specify the number of simultaneous processes to use in the
        simulation with the `number_of_processes` parameter. If
        `number_of_processes` is None (default), the number of processes is
        equal to the number of cores in the machine. To limit the number of
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
        
        with mp.Pool(number_of_processes) as p:
            p.map(
                simulate_single_culture,
                [
                    (k, i, j, self)
                    for k in range(len(self.prob_diff))
                    for i in range(len(self.prob_stem))
                    for j in range(self.num_of_realizations)
                ],
            )


def simulate_single_culture(args: Tuple[int, int, int, Simulation]) -> None:
    """
    A worker function for multiprocessing.

    This function is used by the multiprocessing.Pool instance in the
    simulate_parallel method to parallelize the simulation of different
    cultures. This simulates the growth of a single culture with the given
    parameters and persist the data.

    Parameters
    ----------
    args : tuple
        A tuple containing the indices for the self-replication probability,
        differentiation probability, and realization number, and the instance of
        the Simulation class.

    Notes
    -----
    Due to the way multiprocessing works in Python, you can't directly use
    instance methods as workers for multiprocessing. The multiprocessing
    module needs to be able to pickle the target function, and instance
    methods can't be pickled. Therefore, the instance method worker had to be
    refactored to a standalone function (or a static method).
    """
    k, i, j, sim = args
    current_realization_name = (
        f"culture_pd={sim.prob_diff[k]}_ps={sim.prob_stem[i]}_realization_{j}"
    )
    sim.cultures[current_realization_name] = Culture(
        adjacency_threshold=sim.adjacency_threshold,
        cell_radius=sim.cell_radius,
        cell_max_repro_attempts=sim.cell_max_repro_attempts,
        first_cell_is_stem=sim.first_cell_is_stem,
        prob_stem=sim.prob_stem[i],
        prob_diff=sim.prob_diff[k],
        continuous_graph_generation=sim.continuous_graph_generation,
        rng_seed=sim.rng.integers(low=2**20, high=2**50),
    )
    sim.cultures[current_realization_name].simulate_with_persistent_data(
        sim.num_of_steps_per_realization,
        current_realization_name,
    )
