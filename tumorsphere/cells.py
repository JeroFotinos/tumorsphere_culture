"""
Module containing the Cell class used for simulating cells in a culture.

Classes:
    - Cell: Represents a single cell in a culture. Dependent on the Culture
    class.
"""
import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# colors = {True: "red", False: "blue"}

# probabilities = {'ps' : 0.36, 'pd' : 0.16}
# prob_stem = 0.36


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
