"""
Module containing the Cell class used for simulating cells in a culture.

Classes:
    - Cell: Represents a single cell in a culture. Dependent on the Culture
    class.
"""
from typing import Dict, Set, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np


class Cell:
    """Represents a single cell in a culture.

    Parameters
    ----------
    position : numpy.ndarray
        A vector with 3 components representing the position of the cell.
    culture : tumorsphere.Culture
        The culture to which the cell belongs.
    rng : numpy.random.Generator
        The random number generator used by the cell. In most
        cases, this should be left to be managed by the culture.
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

    Attributes
    ----------
    (All parameters, plus the following.)
    _position_index: int
        This cell's index in the culture's position matrix
    _swap_probability : float
        The probability that, after an asymmetrical reproduction, the position
        of the stem cell is the new child position (probability of swapping
        parent and child positions).
    _colors : dict
        It defines the mapping between tuples of boolean values given by
        (is_stem, in_active_cells) and the color to use when plotting.
    neighbors : set
        Contains the set of neighbors, which are cells within a distance
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
        cells in the set.
    get_neighbors_up_to_second_degree()
        Get the set of neighbors up to second degree.
    get_neighbors_up_to_third_degree()
        Returns the set of cells that are neighbors of the current cell up
        to the third degree.
    find_neighbors()
        Find neighboring cells from the neighbors of the current cell up
        to some degree, keeping the current cells in the set.
    find_neighbors_from_scratch()
        Find neighboring cells from the neighbors of the current cell up
        to some degree, re-calculating from scratch.
    generate_new_position()
        Generate a proposed position for the child cell, adjacent to the current one.
    reproduce()
        The cell reproduces, generating a new child cell.
    """

    _position_index: int
    # culture: Culture
    rng: np.random.Generator
    adjacency_threshold: float
    radius: float
    is_stem: bool
    max_repro_attempts: int
    prob_stem: float
    prob_diff: float
    continuous_graph_generation: bool
    _swap_probability: float
    _colors: Dict[Tuple[bool, bool], str]
    neighbors: Set["Cell"]
    available_space: bool

    def __init__(
        self,
        position: np.ndarray,
        culture,  # : Culture,
        rng: np.random.Generator,
        adjacency_threshold: float = 4,  # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        radius: float = 1,
        is_stem: bool = False,
        max_repro_attempts: int = 1000,
        prob_stem: float = 0.36,  # Wang HARD substrate value
        prob_diff: float = 0,
        continuous_graph_generation: bool = False,
    ) -> None:
        # Add this cell to the culture's position matrix
        self._position_index = len(culture.cell_positions)
        culture.cell_positions = np.append(
            culture.cell_positions, [position], axis=0
        )
        # Generic attributes
        self.culture = culture
        self.rng = rng
        self.adjacency_threshold = adjacency_threshold
        self.radius = radius  # radius of cell
        self.max_repro_attempts = max_repro_attempts

        # Stem case attributes
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self._swap_probability = 0.5

        # Plotting and graph related attributes
        self._continuous_graph_generation = continuous_graph_generation
        self._colors = {
            (True, True): "red",
            (True, False): "salmon",
            (False, True): "blue",
            (False, False): "cornflowerblue",
        }  # the tuple is (is_stem, in_active_cells)

        # Attributes that evolve with the simulation
        self.neighbors = set()
        self.available_space = True
        self.is_stem = is_stem

    @property
    def position(self) -> np.ndarray:
        """A vector with 3 components representing the position of the cell."""
        return self.culture.cell_positions[self._position_index]

    def find_neighbors_from_entire_culture_from_scratch(self) -> None:
        """Find neighboring cells from the entire culture, re-calculating from scratch.

        This method clears the current neighbor set and calculates a new neighbor set
        for the current cell, iterating over all cells in the culture. For each cell,
        it checks if it is not the current cell and not already in the neighbor set,
        and if it is within the adjacency threshold. If all conditions are met, the
        cell is added to the neighbor set.

        Returns:
            None
        """
        self.neighbors = set()
        # si las células se mueven, hay que calcular todo el conjunto de cero
        for cell in self.culture.cells:
            neither_self_nor_neighbor: bool = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.add(cell)

    def find_neighbors_from_entire_culture(self) -> None:
        """Find neighboring cells from the entire culture, keeping the current
        cells in the set.

        This method keeps and updates the neighbor set for the current cell,
        by looking at all cells in the culture. For each cell, it checks if
        it is not the current cell and not already in the neighbor set, and
        if it is within the adjacency threshold. If all conditions are met,
        the cell is added to the neighbor set.

        Returns:
            None
        """
        # como las células no se mueven, sólo se pueden agregar vecinos, por
        # lo que no hay necesidad de reiniciar el conjunto, sólo añadimos
        # los posibles nuevos vecinos
        for cell in self.culture.cells:
            neither_self_nor_neighbor: bool = (cell is not self) and (
                cell not in self.neighbors
            )
            in_neighborhood = (
                np.linalg.norm(self.position - cell.position)
                <= self.adjacency_threshold
            )
            to_append = neither_self_nor_neighbor and in_neighborhood
            if to_append:
                self.neighbors.add(cell)

    def get_neighbors_up_to_second_degree(self) -> Set["Cell"]:
        """Get the set of neighbors up to second degree.

        A cell's neighbors up to second degree are defined as the cell's direct
        neighbors and the neighbors of those neighbors, excluding the cell itself.
        This method returns a list of unique cells that meet this criteria.

        Returns
        -------
        Set[Cell]
            A set of `Cell` objects that are neighbors of the current cell up
            to the second degree.
        """
        neighbors_up_to_second_degree: Set[Cell] = set(self.neighbors)
        for cell1 in self.neighbors:
            new_neighbors: Set[Cell] = cell1.neighbors.difference(
                neighbors_up_to_second_degree
            )
            neighbors_up_to_second_degree.update(new_neighbors)
            for cell2 in new_neighbors:
                neighbors_up_to_second_degree.update(cell2.neighbors)
        return neighbors_up_to_second_degree

    def get_neighbors_up_to_third_degree(self) -> Set["Cell"]:
        """Returns the set of cells that are neighbors of the current cell up
        to the third degree.

        This method returns a set of unique cells that are neighbors to the
        cell, or neighbors of neighbors, recurrently up to third degree.

        Returns
        -------
        Set[Cell]
            A set of `Cell` objects that are neighbors of the current cell up
            to the third degree.
        """
        neighbors_up_to_third_degree: Set[Cell] = set(self.neighbors)
        for cell1 in self.neighbors:
            new_neighbors: Set[Cell] = cell1.neighbors.difference(
                neighbors_up_to_third_degree
            )
            neighbors_up_to_third_degree.update(new_neighbors)
            for cell2 in new_neighbors:
                new_neighbors_l2: Set[Cell] = cell2.neighbors.difference(
                    neighbors_up_to_third_degree
                )
                neighbors_up_to_third_degree.update(new_neighbors_l2)
                for cell3 in new_neighbors_l2:
                    neighbors_up_to_third_degree.update(cell3.neighbors)
        return neighbors_up_to_third_degree

    def find_neighbors(self) -> None:
        """Find neighboring cells from the neighbors of the current cell up
        to some degree, keeping the current cells in the set.

        This method keeps and updates the neighbor set for the current cell,
        by looking recursively at neighbors of the cell, up to certain degree.
        For each cell, it checks if it is not the current cell and not already
        in the neighbor set, and if it is within the adjacency threshold. If
        all conditions are met, the cell is added to the neighbor set.
        If the set of neighbors has less than 12 cells, we look up to third
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
                self.get_neighbors_up_to_third_degree()
            )
        else:
            neighbors_up_to_certain_degree = (
                self.get_neighbors_up_to_second_degree()
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
                self.neighbors.add(cell)

    def find_neighbors_from_scratch(self):
        """Find neighboring cells from the neighbors of the current cell up
        to some degree, re-calculating from scratch.

        This method clears and re-calculates the neighbor set for the current
        cell, by looking recursively at neighbors of the cell, up to certain
        degree. For each cell, it checks if it is not the current cell and not
        already in the neighbor set, and if it is within the adjacency
        threshold. If all conditions are met, the cell is added to the
        neighbor set. If the set of neighbors has less than 12 cells, we
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
                self.get_neighbors_up_to_third_degree()
            )
        else:
            neighbors_up_to_certain_degree = (
                self.get_neighbors_up_to_second_degree()
            )
        # we reset the neighbors set
        self.neighbors = set()
        # we add the cells to the set
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
                self.neighbors.add(cell)

    def generate_new_position(self) -> np.ndarray:
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

    def reproduce(self) -> None:
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
                    self.get_neighbors_up_to_second_degree()
                )
                neighbor_indices = [
                    cell._position_index
                    for cell in neighbors_up_to_second_degree
                ]

                # array with the distances from the proposed child position to the other cells
                if len(neighbors_up_to_second_degree) > 0:
                    neighbor_position_mat = self.culture.cell_positions[
                        neighbor_indices, :
                    ]
                    distance = np.linalg.norm(
                        child_position - neighbor_position_mat, axis=1
                    )
                else:
                    distance = np.array([])

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
                            rng=self.rng,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=True,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
                            prob_diff=self.prob_diff,
                            continuous_graph_generation=self._continuous_graph_generation,
                        )
                    else:
                        child_cell = Cell(
                            position=child_position,
                            culture=self.culture,
                            rng=self.rng,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=False,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
                            prob_diff=self.prob_diff,
                            continuous_graph_generation=self._continuous_graph_generation,
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
                        rng=self.rng,
                        adjacency_threshold=self.adjacency_threshold,
                        radius=self.radius,
                        is_stem=False,
                        max_repro_attempts=self.max_repro_attempts,
                        prob_stem=self.prob_stem,
                        prob_diff=self.prob_diff,
                        continuous_graph_generation=self._continuous_graph_generation,
                    )
                # we add this cell to the culture's cells and active_cells lists
                self.culture.cells.append(child_cell)
                self.culture.active_cells.append(child_cell)
                # we add the parent as first neighbor (necessary for
                # the find_neighbors that are not from_entire_culture)
                child_cell.neighbors.add(self)
                # we find the child's neighbors
                child_cell.find_neighbors()
                # we add the child as a neighbor of its neighbors
                for cell in child_cell.neighbors:
                    cell.neighbors.add(child_cell)
                # we add the child to the graph (node and edges)
                if self._continuous_graph_generation:
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


# ============================================================================
#
# ------------------------------- LITE VERSION -------------------------------
#
# ============================================================================


@dataclass(frozen=False, slots=True)
class CellLite:
    """Represents a single cell in a Culture.

    Attributes
    ----------
    culture: CultureLite
        The culture to which the cell belongs.
    is_stem: bool
        Whether the cell is a stem cell or not.
    parent_index: Optional[int]
        The index of the parent cell in the culture's cell_positions array.
        Default is 0.
    neighbors_indexes: Set[int]
        A set of indexes corresponding to the neighboring cells in the
        culture's cell_positions array. Default is an empty set.
    available_space: bool
        Whether the cell has available space around it or not. Default is True.
    _position_index: Optional[int]
        The index of the cell's position in the culture's cell_positions array.
        It's not directly settable during instantiation.

    Methods
    -------
    __init__(position, culture, is_stem, parent_index=0, neighbors_indexes=set(), available_space=True)
        Initializes the CellLite object and sets the _position_index attribute
        based on the position given.

    """

    culture: "CultureLite"
    is_stem: bool
    parent_index: Optional[int] = 0
    neighbors_indexes: Set[int] = field(default_factory=set)
    available_space: bool = True
    _position_index: Optional[int] = field(default=False, init=False)

    def __init__(
        self,
        position: np.ndarray,
        culture: "CultureLite",
        is_stem: bool,
        parent_index: Optional[int] = 0,
        available_space: bool = True,  # not to be set by user
    ) -> None:
        """
        Initializes the CellLite object.

        Parameters
        ----------
        position : np.ndarray
            The position of the cell. This is used to update the cell_positions in the culture and
            set the _position_index attribute, but is not stored as an attribute in the object itself.
        culture : CultureLite
            The culture to which the cell belongs.
        is_stem : bool
            Whether the cell is a stem cell or not.
        parent_index : Optional[int], default=0
            The index of the parent cell in the culture's cell_positions array.
        neighbors_indexes : Set[int], default=set()
            A set of indexes corresponding to the neighboring cells in the culture's cell_positions array.
        available_space : bool, default=True
            Whether the cell has available space around it or not.

        """
        self.culture = culture
        self.is_stem = is_stem
        self.parent_index = parent_index
        self.neighbors_indexes = set()
        self.available_space = available_space

        # we FIRST get the cell's index
        self._position_index = len(culture.cell_positions)

        # and THEN add the cell to the culture's position matrix and cell
        # lists, in the previous index
        culture.cell_positions = np.append(
            culture.cell_positions, [position], axis=0
        )
        self.culture.cells.append(self)
        self.culture.active_cells.append(self._position_index)
