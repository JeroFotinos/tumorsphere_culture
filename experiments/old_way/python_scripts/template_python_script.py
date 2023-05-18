"""Template of python script to launch many realizations, for different
parameter combinations, of the simulation of the growth of a tumorsphere.
This file is modified by a bash script that submits the job to the queue.
"""


# ========================================================================
# CELLS.PY
# ========================================================================


import copy
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Cell:
    """Represents a single cell in a culture.

    Parameters
    ----------
    position : numpy.ndarray
        A vector with 3 components representing the position of the cell.
    culture : tumorsphere.Culture
        The culture to which the cell belongs.
    adjacency_threshold : float, optional
        The maximum distance between cells for them to be considered neighbors.
        Defaults to 4, but is inherited from cell to cell (so the first cell
        dictates the value for the rest).
    radius : float, optional
        The radius of the cell. Defaults to 1, but is inherited from cell to cell.
    is_stem : bool, optional
        True if the cell is a stem cell, False otherwise. Defaults to False,
        but its value is managed by the parent cell.
    max_repro_attempts : int, optional
        The maximum number of times the cell will attempt to reproduce before
        becoming inactive. Defaults to 1000, but is inherited from cell to cell.
    prob_stem : float, optional
        The probability that a stem cell will self-replicate. Defaults to 0.36
        for being the value measured by Benítez et al. (BMC Cancer, (2021),
        1-11, 21(1))for the experiment of Wang et al. (Oncology Letters,
        (2016), 1355-1360, 12(2)) on a hard substrate. Nevertheless, it is
        inherited from cell to cell.
    prob_diff : float
        The probability that a stem cell will yield a differentiated cell.
        Defaults to 0 (because the intention was to see if percolation occurs,
        and if it doesn't happen at prob_diff = 0, it will never happen).
    continuous_graph_generation : bool
        True if the cell should continuously generate a graph of its neighbors, False otherwise.
    rng_seed : int, optional
        The seed for the random number generator used by the cell. In most
        cases, this should be left to be managed by the parent cell.

    Attributes
    ----------
    (All parameters, plus the following.)
    _swap_probability : float
        The probability that, after an asymmetrical reproduction, the position
        of the stem cell is the new child position (probability of swapping
        parent and child positions).
    _colors : dict
        It defines the mapping between tuples of boolean values given by
        (is_stem, in_active_cells) and the color to use when plotting.
    neighbors : list
        Contains the list of neighbors, which are cells within a distance
        equal or less than adjacency_threshold.
    available_space : bool, default=True
        Specify whether the cell is considered active. It corresponds to
        whether the cell has had a time step where it failed to reproduce.


    Methods
    -------
    find_neighbors_from_entire_culture_from_scratch()
        Find neighboring cells from the entire culture, re-calculating from
        scratch.
    find_neighbors_from_entire_culture()
        Find neighboring cells from the entire culture, keeping the current
        cells in the list.
    get_list_of_neighbors_up_to_second_degree()
        Get a list of neighbors up to second degree.
    get_list_of_neighbors_up_to_third_degree()
        Returns a list of cells that are neighbors of the current cell up
        to the third degree.
    find_neighbors()
        Find neighboring cells from the neighbors of the current cell up
        to some degree, keeping the current cells in the list.
    find_neighbors_from_scratch()
        Find neighboring cells from the neighbors of the current cell up
        to some degree, re-calculating from scratch.
    generate_new_position()
        Generate a proposed position for the child cell, adjacent to the current one.
    reproduce()
        The cell reproduces, generating a new child cell.
    """

    def __init__(
        self,
        position,
        culture,
        adjacency_threshold=4,  # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        radius=1,
        is_stem=False,
        max_repro_attempts=1000,
        prob_stem=0.36,  # Wang HARD substrate value
        prob_diff=0,
        continuous_graph_generation=False,
        rng_seed=23978461273864
        # THE CULTURE MUST PROVIDE A SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # Generic attributes
        self.position = position  # NumPy array, vector with 3 components
        self.culture = culture
        self.adjacency_threshold = adjacency_threshold
        self.radius = radius  # radius of cell
        self.max_repro_attempts = max_repro_attempts

        # Stem case attributes
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self._swap_probability = 0.5

        # We instantiate the cell's RNG with the entropy provided
        self.rng = np.random.default_rng(rng_seed)

        # Plotting and graph related attributes
        self._continuous_graph_generation = continuous_graph_generation
        self._colors = {
            (True, True): "red",
            (True, False): "salmon",
            (False, True): "blue",
            (False, False): "cornflowerblue",
        }  # the tuple is (is_stem, in_active_cells)

        # Attributes that evolve with the simulation
        self.neighbors = []
        self.available_space = True
        self.is_stem = is_stem

    def find_neighbors_from_entire_culture_from_scratch(self):
        """Find neighboring cells from the entire culture, re-calculating from scratch.

        This method clears the current neighbor list and calculates a new neighbor list
        for the current cell, iterating over all cells in the culture. For each cell,
        it checks if it is not the current cell and not already in the neighbor list,
        and if it is within the adjacency threshold. If all conditions are met, the
        cell is added to the neighbor list.

        Returns:
            None
        """
        self.neighbors = []
        # si las células se mueven, hay que calcular toda la lista de cero
        for cell in self.culture.cells:
            neither_self_nor_neighbor = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.append(cell)

    def find_neighbors_from_entire_culture(self):
        """Find neighboring cells from the entire culture, keeping the current
        cells in the list.

        This method keeps and updates the neighbor list for the current cell,
        by looking at all cells in the culture. For each cell, it checks if
        it is not the current cell and not already in the neighbor list, and
        if it is within the adjacency threshold. If all conditions are met,
        the cell is added to the neighbor list.

        Returns:
            None
        """
        # como las células no se mueven, sólo se pueden agregar vecinos, por
        # lo que no hay necesidad de reiniciar la lista, sólo añadimos
        # los posibles nuevos vecinos
        for cell in self.culture.cells:
            neither_self_nor_neighbor = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.append(cell)

    def get_list_of_neighbors_up_to_second_degree(self):
        """Get a list of neighbors up to second degree.

        A cell's neighbors up to second degree are defined as the cell's direct
        neighbors and the neighbors of those neighbors, excluding the cell itself.
        This method returns a list of unique cells that meet this criteria.

        Returns
        -------
        List[Cell]
            A list of `Cell` objects that are neighbors of the current cell up
            to the second degree.
        """
        neighbors_up_to_second_degree = set(self.neighbors)
        for cell1 in self.neighbors:
            neighbors_up_to_second_degree = (
                neighbors_up_to_second_degree.union(set(cell1.neighbors))
            )
            for cell2 in cell1.neighbors:
                neighbors_up_to_second_degree = (
                    neighbors_up_to_second_degree.union(set(cell2.neighbors))
                )
        neighbors_up_to_second_degree = list(neighbors_up_to_second_degree)
        return neighbors_up_to_second_degree

    def get_list_of_neighbors_up_to_third_degree(self):
        """Returns a list of cells that are neighbors of the current cell up
        to the third degree.

        This method returns a list of unique cells that are neighbors to the
        cell, or neighbors of neighbors, recurrently up to third degree.

        Returns
        -------
        List[Cell]
            A list of `Cell` objects that are neighbors of the current cell up
            to the third degree.
        """
        neighbors_up_to_third_degree = set(self.neighbors)
        for cell1 in self.neighbors:
            neighbors_up_to_third_degree = neighbors_up_to_third_degree.union(
                set(cell1.neighbors)
            )
            for cell2 in cell1.neighbors:
                neighbors_up_to_third_degree = (
                    neighbors_up_to_third_degree.union(set(cell2.neighbors))
                )
                for cell3 in cell2.neighbors:
                    neighbors_up_to_third_degree = (
                        neighbors_up_to_third_degree.union(
                            set(cell3.neighbors)
                        )
                    )
        neighbors_up_to_third_degree = list(neighbors_up_to_third_degree)
        return neighbors_up_to_third_degree

    def find_neighbors(self):
        """Find neighboring cells from the neighbors of the current cell up
        to some degree, keeping the current cells in the list.

        This method keeps and updates the neighbor list for the current cell,
        by looking recursively at neighbors of the cell, up to certain degree.
        For each cell, it checks if it is not the current cell and not already
        in the neighbor list, and if it is within the adjacency threshold. If
        all conditions are met, the cell is added to the neighbor list.
        If the list of neighbors has less than 12 cells, we look up to third
        neighbors, if not, just up to second neighbors. This decision stems
        from the fact that if the cell is a newborn, it will only have its
        parent as a neighbor, so the neighbors of its neighbors are just the
        neighbors of its parent. Therefore, the first time we have to go a
        level deeper.

        Returns:
            None
        """
        if len(self.neighbors) < 12:
            neighbors_up_to_certain_degree = (
                self.get_list_of_neighbors_up_to_third_degree()
            )
        else:
            neighbors_up_to_certain_degree = (
                self.get_list_of_neighbors_up_to_second_degree()
            )
        # now we check if there are cells to append
        for cell in neighbors_up_to_certain_degree:
            neither_self_nor_neighbor = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.append(cell)

    def find_neighbors_from_scratch(self):
        """Find neighboring cells from the neighbors of the current cell up
        to some degree, re-calculating from scratch.

        This method clears and re-calculates the neighbor list for the current
        cell, by looking recursively at neighbors of the cell, up to certain
        degree. For each cell, it checks if it is not the current cell and not
        already in the neighbor list, and if it is within the adjacency
        threshold. If all conditions are met, the cell is added to the
        neighbor list. If the list of neighbors has less than 12 cells, we
        look up to third neighbors, if not, just up to second neighbors. This
        decision stems from the fact that if the cell is a newborn, it will
        only have its parent as a neighbor, so the neighbors of its neighbors
        are just the neighbors of its parent. Therefore, the first time we
        have to go a level deeper.

        Returns:
            None
        """
        if len(self.neighbors) < 20:
            neighbors_up_to_certain_degree = (
                self.get_list_of_neighbors_up_to_third_degree()
            )
        else:
            neighbors_up_to_certain_degree = (
                self.get_list_of_neighbors_up_to_second_degree()
            )
        # we reset the neighbors list
        self.neighbors = []
        # we add the cells to the list
        for cell in neighbors_up_to_certain_degree:
            neither_self_nor_neighbor = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.append(cell)

    def generate_new_position(self):
        """Generate a proposed position for the child cell, adjacent to the current one.

        A new position for the child cell is randomly generated, at a distance
        equals to two times the radius of a cell (all cells are assumed to
        have the same radius) by randomly choosing the angular spherical
        coordinates from a uniform distribution. It uses the cell current
        position and its radius.

        Returns
        -------
        new_position : numpy.ndarray
            A 3D vector representing the new position of the cell.
        """
        theta = np.random.uniform(low=0, high=2 * np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        x = 2 * self.radius * np.sin(phi) * np.cos(theta)
        y = 2 * self.radius * np.sin(phi) * np.sin(theta)
        z = 2 * self.radius * np.cos(phi)
        new_position = self.position + np.array([x, y, z])
        return new_position

    def reproduce(self):
        """The cell reproduces, generating a new child cell.

        Attempts to create a new cell in a random position, adjacent to the
        current cell, if the cell has available space. If the cell fails to
        find a position that doesn't overlap with existing cells, for the
        estabished maximum number of attempts, no new cell is created.

        Raises:
            AssertionError: If the number of neighbors exceeds the number of
            cells in the culture.

        Returns:
            None.
        """
        assert len(self.neighbors) <= len(self.culture.cells)

        if self.available_space:
            for attempt in range(self.max_repro_attempts):
                child_position = self.generate_new_position()
                neighbors_up_to_second_degree = (
                    self.get_list_of_neighbors_up_to_second_degree()
                )
                # array with the distances from the proposed child position to the other cells
                distance = np.array(
                    [
                        np.linalg.norm(child_position - cell.position)
                        for cell in neighbors_up_to_second_degree
                    ]
                )
                # boolean array specifying if there is no overlap between
                # the proposed child position and the other cells
                no_overlap = np.all(distance >= 2 * self.radius)
                # if it is true that there is no overlap for
                # every element of the array, we break the loop
                if no_overlap:
                    break

            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available space
            if no_overlap:
                # we create a child in that position
                if self.is_stem:
                    random_number = self.rng.random()
                    if random_number <= self.prob_stem:  # ps
                        child_cell = Cell(
                            position=child_position,
                            culture=self.culture,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=True,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
                            prob_diff=self.prob_diff,
                            continuous_graph_generation=self._continuous_graph_generation,
                            rng_seed=self.rng.integers(
                                low=2**20, high=2**50
                            ),
                        )
                    else:
                        child_cell = Cell(
                            position=child_position,
                            culture=self.culture,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=False,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
                            prob_diff=self.prob_diff,
                            continuous_graph_generation=self._continuous_graph_generation,
                            rng_seed=self.rng.integers(
                                low=2**20, high=2**50
                            ),
                        )
                        if random_number <= (
                            self.prob_stem + self.prob_diff
                        ):  # pd
                            self.is_stem = False
                        elif (
                            self.rng.random() <= self._swap_probability
                        ):  # pa = 1-ps-pd
                            self.is_stem = False
                            child_cell.is_stem = True
                else:
                    child_cell = Cell(
                        position=child_position,
                        culture=self.culture,
                        adjacency_threshold=self.adjacency_threshold,
                        radius=self.radius,
                        is_stem=False,
                        max_repro_attempts=self.max_repro_attempts,
                        prob_stem=self.prob_stem,
                        prob_diff=self.prob_diff,
                        continuous_graph_generation=self._continuous_graph_generation,
                        rng_seed=self.rng.integers(low=2**20, high=2**50),
                    )
                # we add this cell to the culture's cells and active_cells lists
                self.culture.cells.append(child_cell)
                self.culture.active_cells.append(child_cell)
                # we add the parent as first neighbor (necessary for
                # the find_neighbors that are not from_entire_culture)
                child_cell.neighbors.append(self)
                # we find the child's neighbors
                child_cell.find_neighbors()
                # we add the child as a neighbor of its neighbors
                for cell in child_cell.neighbors:
                    cell.neighbors.append(child_cell)
                # we add the child to the graph (node and edges)
                if self._continuous_graph_generation == True:
                    self.culture.graph.add_node(child_cell)
                    for cell in child_cell.neighbors:
                        self.culture.graph.add_edge(child_cell, cell)
            else:
                self.available_space = False
                self.culture.active_cells.remove(self)
                # if there was no available space, we turn off reproduction
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing (reproduction is turned off)


# ========================================================================
# CULTURE.PY
# ========================================================================


class Culture:
    """Class that represents a culture of cells.

    It contains methods to manipulate the cells and the graph representing
    the culture. The culture can be visualized using the provided plotting
    methods.

    Parameters
    ----------
    adjacency_threshold : float, optional
        The maximum distance between two cells for them to be considered
        neighbors. Default is 4, which is an upper bound to 2 * sqrt(2)
        (the second neighbor distance in a hexagonal close-packed lattice,
        which is the high density limit case).
    cell_radius : float, optional
        The radius of the cells in the culture. Default is 1.
    cell_max_repro_attempts : int, optional
        The maximum number of attempts that a cell will make to reproduce before
        giving up, setting it available_space attribute to false, and removing
        itself from the list of active cells. Default is 1000.
    first_cell_is_stem : bool, optional
        Whether the first cell in the culture is a stem cell. Default is False.
    prob_stem : float, optional
        The probability that a stem cell will self-replicate. Defaults to 0.36
        for being the value measured by Benítez et al. (BMC Cancer, (2021),
        1-11, 21(1))for the experiment of Wang et al. (Oncology Letters,
        (2016), 1355-1360, 12(2)) on a hard substrate.
    prob_diff : float, optional
        The probability that a stem cell will yield a differentiated cell.
        Defaults to 0 (because the intention was to see if percolation occurs,
        and if it doesn't happen at prob_diff = 0, it will never happen).
    continuous_graph_generation : bool, optional
        Whether the graph representing the culture should be continuously updated
        as cells are added. Default is False.
    rng_seed : int, optional
        The seed to be used by the culture's random number generator. Default is
        110293658491283598. Nevertheless, this should be managed by the
        Simulation object.

    Attributes
    ----------
    (All parameters, plus the following.)
    rng : numpy.random.Generator
        The culture's random number generator.
    cells : list of Cell
        The list of all cells in the culture.
    active_cells : list of Cell
        The list of all active cells in the culture, i.e., cells that still
        have its available_space attribute set to True and can still reproduce.
    graph : networkx.Graph
        The graph representing the culture. Nodes are cells and edges represent
        the adjacency relationship between cells.

    Methods
    -------
    plot_culture_dots()
        Plot the cells in the culture as dots in a 3D scatter plot.
    plot_culture_spheres()
        Plot the cells in the culture as spheres in a 3D scatter plot.
    plot_culture_fig()
        Plot the cells in the culture as spheres in a 3D scatter plot and
        return the figure object.
    plot_graph(self)
        Plot the cell graph using networkx.
    generate_adjacency_graph_from_scratch()
        Re-generates the culture graph containing cells and their neighbors
        using NetworkX.
    simulate(num_times)
        Simulates cell reproduction for a given number of time steps.
    any_csc_in_culture_boundary()
        Check if there is any cancer stem cell (CSC) in the boundary of the
        culture.
    simulate_with_data(num_times)
        Simulate culture growth for a specified number of time steps and
        record the data at each time step.
    """

    def __init__(
        self,
        adjacency_threshold=4,  # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        cell_radius=1,
        cell_max_repro_attempts=1000,
        first_cell_is_stem=False,
        prob_stem=0.36,  # Wang HARD substrate value
        prob_diff=0,
        continuous_graph_generation=False,
        rng_seed=110293658491283598
        # THE SIMULATION MUST PROVIDE A SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # attributes to inherit to the cells
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff

        # we instantiate the culture's RNG with the entropy provided
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # we instantiate the first cell
        first_cell_object = Cell(
            position=np.array([0, 0, 0]),
            culture=self,
            adjacency_threshold=self.adjacency_threshold,
            radius=self.cell_radius,
            is_stem=self.first_cell_is_stem,
            max_repro_attempts=cell_max_repro_attempts,
            prob_stem=self.prob_stem,
            prob_diff=self.prob_diff,
            continuous_graph_generation=continuous_graph_generation,
            rng_seed=self.rng.integers(low=2**20, high=2**50),
        )

        # we initialize the lists and graphs with the first cell
        self.cells = [first_cell_object]
        self.active_cells = [first_cell_object]
        self.graph = nx.Graph()
        self.graph.add_node(first_cell_object)

    # ========================= Ploting methods ==========================

    def plot_culture_dots(self):
        """Plot the cells in the culture as dots in a 3D scatter plot."""
        positions = np.array(
            [self.cells[i].position for i in range(len(self.cells))]
        )
        # colors = np.array([cell.culture for cell in self.cells])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2], c=(0, 1, 0)
        )  # color = green in RGB
        plt.show()

    def plot_culture_spheres(self):
        """Plot the cells in the culture as spheres in a 3D scatter plot."""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for cell in self.cells:
            x, y, z = cell.position
            ax.scatter(
                x,
                y,
                z,
                c=cell._colors[(cell.is_stem, cell in self.active_cells)],
                marker="o",
            )

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=cell._colors[(cell.is_stem, cell in self.active_cells)],
                alpha=0.2,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.mouse_init()  # initialize mouse rotation

        plt.show()

    def plot_culture_fig(self):
        """Plot the cells in the culture as spheres in a 3D scatter plot and
        return the figure object.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object of the plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for cell in self.cells:
            x, y, z = cell.position
            ax.scatter(
                x,
                y,
                z,
                c=cell._colors[(cell.is_stem, cell in self.active_cells)],
                marker="o",
            )

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=cell._colors[(cell.is_stem, cell in self.active_cells)],
                alpha=0.2,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ax.mouse_init()  # initialize mouse rotation

        # plt.show()
        return fig

    def plot_graph(self):
        """Plot the cell graph using networkx."""
        nx.draw(self.graph)

    # to be implemented

    def generate_adjacency_graph_from_scratch(self):
        """Re-generates the culture graph containing cells and their neighbors
        using NetworkX.
        """
        self.graph = nx.Graph()
        for cell in self.cells:
            self.graph.add_node(cell)
        for i, cell1 in enumerate(self.cells):
            for cell2 in self.cells[i + 1 :]:
                if cell2 in cell1.neighbors:
                    self.graph.add_edge(cell1, cell2)

    # ====================================================================

    def simulate(self, num_times):
        """Simulates cell reproduction for a given number of time steps.

        Parameters
        ----------
        num_times : int
            Number of time steps to simulate.
        """
        for i in range(num_times):
            cells = self.rng.permutation(self.active_cells)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for cell in cells:
                cell.reproduce()

    def any_csc_in_culture_boundary(self):
        """Check if there is any cancer stem cell (CSC) in the boundary of the
        culture.

        Returns
        -------
        bool
            Whether or not there is any CSC in the culture's list of active
            cells.
        """
        stem_in_boundary = [
            (cell.available_space and cell.is_stem)
            for cell in self.active_cells
        ]
        any_csc_in_boundary = np.any(stem_in_boundary)
        return any_csc_in_boundary

    def simulate_with_data(self, num_times):
        """Simulate culture growth for a specified number of time steps and
        record the data at each time step.

        At each time step, we randomly sort the list of active cells and then
        we tell them to reproduce one by one. The data that gets recorded at
        each step is the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.

        Returns
        -------
        dict
            A dictionary with keys representing the different types of data that
            were recorded and values representing the recorded data at each time
            step, in the form of numpy.array's representing the time series of
            the data. The types of data recorded are 'total', 'active',
            'total_stem', and 'active_stem'.
        """
        # we use a dictionary to store the data arrays and initialize them
        data = {
            "total": np.zeros(num_times),
            "active": np.zeros(num_times),
            "total_stem": np.zeros(num_times),
            "active_stem": np.zeros(num_times),
        }

        # we count the initial amount of CSCs
        if self.first_cell_is_stem:
            initial_amount_of_csc = 1
        else:
            initial_amount_of_csc = 0

        # we asign the initial values for the data
        data["total"][0] = 1
        data["active"][0] = 1
        data["total_stem"][0] = initial_amount_of_csc
        data["active_stem"][0] = initial_amount_of_csc

        # we simulate for num_times time steps
        for i in range(1, num_times):
            # we get a permuted copy of the cells list
            cells = self.rng.permutation(self.active_cells)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for cell in cells:
                cell.reproduce()

            # we count the number of CSCs in this time step
            total_stem_counter = 0
            for cell in self.cells:
                if cell.is_stem:
                    total_stem_counter = total_stem_counter + 1

            # we count the number of active CSCs in this time step
            active_stem_counter = 0
            for cell in self.active_cells:
                if cell.is_stem:
                    active_stem_counter = active_stem_counter + 1

            # we asign the data values for this time step
            data["total"][i] = len(self.cells)
            data["active"][i] = len(self.active_cells)
            data["total_stem"][i] = total_stem_counter
            data["active_stem"][i] = active_stem_counter

        return data

    # def simulate_with_continuos_data
    # def simulate_with_continuos_data_and_persisted_culture


# ========================================================================
# SIMULATION.PY
# ========================================================================


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
        the seed on which every other seed depends. Default is
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
        Runs the simulation.
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

    def simulate(self):
        """Simulate the culture growth for different self-replication and
        differentiation probabilities and realizations and compute the average
        data for each of the self-replication and differentiation probability
        combinations.
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

        # picklear los objetos tiene que ser responsabilidad del método
        # simulate de culture, ya que es algo que se hace en medio de la
        # evolución, pero va a necesitar que le pase el current_realization_name
        # para usarlo como nombre del archivo

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

    # la idea es usar esto haciendo
    # fig, ax = simulation.plot_average_data()
    # plt.show()


# ========================================================================
# SCRIPT.PY
# ========================================================================


sim = Simulation(
    first_cell_is_stem=True,
    prob_stem=[val_ps],  # Wang HARD substrate value
    num_of_realizations=1,
    num_of_steps_per_realization=60,
    rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
    cell_radius=1,
    adjacency_threshold=4,
    cell_max_repro_attempts=1000,
    continuous_graph_generation=False,
)

sim.simulate()
