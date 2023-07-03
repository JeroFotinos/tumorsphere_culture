"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""
import time
from typing import Dict, Set, Tuple, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from tumorsphere.cells import Cell, CellLite


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
    measure_time : bool, optional
        Whether to measure the time it takes to simulate one time step.
        Default is False. This is used for performance testing purposes.
        Results are saved to a file called time_measurements.dat.

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
    cell_positions : numpy.ndarray
        Matrix with the positions of each cell in the culture in each row.
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
        record the data in a dictionary at each time step.
    simulate_with_persistent_data(num_times, culture_name)
        Simulate culture growth for a specified number of time steps and
        persist the data in a file at each time step.

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
        rng_seed=110293658491283598,
        measure_time=False,
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

        self.cell_positions = np.empty((0, 3), float)

        # we instantiate the first cell
        first_cell_object = Cell(
            position=np.array([0, 0, 0]),
            culture=self,
            rng=self.rng,
            adjacency_threshold=self.adjacency_threshold,
            radius=self.cell_radius,
            is_stem=self.first_cell_is_stem,
            max_repro_attempts=cell_max_repro_attempts,
            prob_stem=self.prob_stem,
            prob_diff=self.prob_diff,
            continuous_graph_generation=continuous_graph_generation,
        )

        # we initialize the lists and graphs with the first cell
        self.cells = [first_cell_object]
        self.active_cells = [first_cell_object]
        self.graph = nx.Graph()
        self.graph.add_node(first_cell_object)

        # other attributes
        self.measure_time = measure_time
        # it seems more appropriate for this to be a parameter of the simulate
        # method to be used

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

    def simulate_with_persistent_data(self, num_times, culture_name):
        """Simulate culture growth for a specified number of time steps and
        record the data in a file at each time step, so its available in real
        time.

        At each time step, we randomly sort the list of active cells and then
        we tell them to reproduce one by one. The data that gets recorded at
        each step is the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells. It no
        longer saves data in a dictionary, but rather to a file with a
        specified name.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.
        culture_name : str
            The name of the culture in the simulation, in the format
            culture_pd={prob_diff}_ps={prob_stem}_realization_{j}.dat

        Returns
        -------
        dict
            A dictionary with keys representing the different types of data that
            were recorded and values representing the recorded data at each time
            step, in the form of numpy.array's representing the time series of
            the data. The types of data recorded are 'total', 'active',
            'total_stem', and 'active_stem'.
        """

        # we count the initial amount of CSCs
        if self.first_cell_is_stem:
            initial_amount_of_csc = 1
        else:
            initial_amount_of_csc = 0

        # we write the header and the data values for this time step
        with open(f"data/{culture_name}.dat", "w") as file:
            file.write("total, active, total_stem, active_stem \n")
            file.write(
                f"1, 1, {initial_amount_of_csc}, {initial_amount_of_csc} \n"
            )

        # we simulate for num_times time steps
        for i in range(1, num_times):
            if self.measure_time:
                # For measuring the time it takes to simulate one time step
                # start the timer
                start_time = time.perf_counter()

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

            # we save the data to a file
            with open(f"data/{culture_name}.dat", "a") as file:
                file.write(
                    f"{len(self.cells)}, {len(self.active_cells)}, {total_stem_counter}, {active_stem_counter} \n"
                )

            if self.measure_time:
                # For measuring the time it takes to simulate one time step
                # stop the timer
                end_time = time.perf_counter()

                # calculate the elapsed time
                elapsed_time = end_time - start_time

                # print the elapsed time
                with open("time_per_step.dat", "a") as file:
                    file.write(f"{elapsed_time:.6f} seconds" + "\n")


# ============================================================================
#
# ------------------------------- LITE VERSION -------------------------------
#
# ============================================================================


class CultureLite:
    def __init__(
        self,
        adjacency_threshold: float = 4,
        cell_radius: float = 1,
        cell_max_repro_attempts: int = 1000,
        first_cell_is_stem: bool = True,
        prob_stem: float = 0,
        prob_diff: float = 0,
        rng_seed: int = 110293658491283598,
    ):
        """
        Initialize a new culture of cells.

        Parameters
        ----------
        adjacency_threshold : int, optional
            The maximum distance at which two cells can be considered neighbors,
            by default 4.
        cell_radius : int, optional
            The radius of a cell, by default 1.
        cell_max_repro_attempts : int, optional
            The maximum number of reproduction attempts a cell can make,
            by default 1000.
        first_cell_is_stem : bool, optional
            Whether the first cell is a stem cell or not, by default False.
        prob_stem : float, optional
            The probability that a cell becomes a stem cell, by default 0.
        prob_diff : float, optional
            The probability that a cell differentiates, by default 0.
        rng_seed : int, optional
            Seed for the random number generator, by default 110293658491283598.

        Attributes
        ----------
        cell_max_repro_attempts : int
            Maximum number of reproduction attempts a cell can make.
        adjacency_threshold : int
            The maximum distance at which two cells can be considered neighbors.
        cell_radius : int
            The radius of a cell.
        prob_stem : float
            The probability that a cell becomes a stem cell.
        prob_diff : float
            The probability that a cell differentiates.
        _swap_probability : float
            The probability that a cell swaps its type with its offspring.
        rng : numpy.random.Generator
            Random number generator.
        first_cell_is_stem : bool
            Whether the first cell is a stem cell or not.
        cell_positions : numpy.ndarray
            Matrix to store the positions of all cells in the culture.
        cells : list[CellLite]
            List of all cells in the culture.
        active_cells : list[CellLite]
            List of all active cells in the culture.
        """

        # cell attributes
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self._swap_probability = 0.5

        # we instantiate the culture's RNG with the entropy provided
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # initialize the positions matrix
        self.cell_positions = np.empty((0, 3), float)

        # we initialize the lists of cells
        self.cells = []
        self.active_cells = []

        # we instantiate the first cell
        first_cell_object = CellLite(
            position=np.array([0, 0, 0]),
            culture=self,
            is_stem=self.first_cell_is_stem,
            parent_index=0,
            available_space=True,
        )

    # ------------------cell related behavior------------------

    def get_neighbors_up_to_second_degree(self, cell_index: int) -> Set[int]:
        """
        Get the neighbors up to the second degree of a cell.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Returns
        -------
        Set[int]
            The set of indices of the cell's neighbors up to the second degree.
        """
        cell = self.cells[cell_index]
        neighbors_up_to_second_degree: Set[int] = set(cell.neighbors_indexes)
        for index1 in cell.neighbors_indexes:
            cell1 = self.cells[index1]
            new_neighbors: Set[int] = cell1.neighbors_indexes.difference(
                neighbors_up_to_second_degree
            )
            neighbors_up_to_second_degree.update(new_neighbors)
            for index2 in new_neighbors:
                cell2 = self.cells[index2]
                neighbors_up_to_second_degree.update(cell2.neighbors_indexes)
        return neighbors_up_to_second_degree

    def get_neighbors_up_to_third_degree(self, cell_index: int) -> Set[int]:
        """
        Get the neighbors up to the third degree of a cell.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Returns
        -------
        Set[int]
            The set of indices of the cell's neighbors up to the third degree.
        """
        cell = self.cells[cell_index]
        neighbors_up_to_third_degree: Set[int] = set(cell.neighbors_indexes)
        for index1 in cell.neighbors_indexes:
            cell1 = self.cells[index1]
            new_neighbors: Set[int] = cell1.neighbors_indexes.difference(
                neighbors_up_to_third_degree
            )
            neighbors_up_to_third_degree.update(new_neighbors)
            for index2 in new_neighbors:
                cell2 = self.cells[index2]
                new_neighbors_l2: Set[
                    int
                ] = cell2.neighbors_indexes.difference(
                    neighbors_up_to_third_degree
                )
                neighbors_up_to_third_degree.update(new_neighbors_l2)
                for index3 in new_neighbors_l2:
                    cell3 = self.cells[index3]
                    neighbors_up_to_third_degree.update(
                        cell3.neighbors_indexes
                    )
        return neighbors_up_to_third_degree

    def find_neighbors(self, cell_index: int) -> None:
        """
        Find the neighbors of a cell.

        This method modifies the cell's neighbors attribute in-place.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        """
        cell = self.cells[cell_index]
        if len(cell.neighbors_indexes) < 12:
            neighbors_up_to_certain_degree = (
                self.get_neighbors_up_to_third_degree(cell_index)
            )
        else:
            neighbors_up_to_certain_degree = (
                self.get_neighbors_up_to_second_degree(cell_index)
            )
        # now we check if there are cells to append
        for index in neighbors_up_to_certain_degree:
            a_cell = self.cells[index]
            neither_self_nor_neighbor = (index != cell_index) and (
                index not in cell.neighbors_indexes
            )
            if neither_self_nor_neighbor:
                # if the distance to this cell is within the threshold, we add it as a neighbor
                # distance = np.linalg.norm(
                #     cell.position - a_cell.position
                # )
                distance = np.linalg.norm(
                    self.cell_positions[cell_index]
                    - self.cell_positions[a_cell._position_index]
                )
                in_neighborhood = distance <= self.adjacency_threshold
                if in_neighborhood:
                    cell.neighbors_indexes.add(index)
                    a_cell.neighbors_indexes.add(cell_index)

    def generate_new_position(self, cell_index: int) -> np.ndarray:
        """Generate a proposed position for the child cell, adjacent to the
        given one.

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
        x = 2 * self.cell_radius * np.sin(phi) * np.cos(theta)
        y = 2 * self.cell_radius * np.sin(phi) * np.sin(theta)
        z = 2 * self.cell_radius * np.cos(phi)
        cell_position = self.cell_positions[cell_index]
        new_position = cell_position + np.array([x, y, z])
        return new_position

    def reproduce(
        self, cell_index: int
    ) -> None:  ##### ANTES ERA UN CELL; chequear que no haya cell.position
        """The given cell reproduces, generating a new child cell.

        Attempts to create a new cell in a random position, adjacent to the
        current cell, if the cell has available space. If the cell fails to
        find a position that doesn't overlap with existing cells, for the
        estabished maximum number of attempts, no new cell is created.
        """
        # assert len(cell.neighbors) <= len(self.cells)

        cell = self.cells[cell_index]

        if cell.available_space:
            for attempt in range(self.cell_max_repro_attempts):
                # we generate a new proposed position for the child cell
                child_position = self.generate_new_position(cell_index)

                # we update and get the neighbors set
                self.find_neighbors(cell_index)
                neighbors_up_to_some_degree = cell.neighbors_indexes

                # array with the indices of the neighbors
                neighbor_indices = list(neighbors_up_to_some_degree)

                # array with the distances from the proposed child position to
                # the other cells
                if len(neighbors_up_to_some_degree) > 0:
                    neighbor_position_mat = self.cell_positions[
                        neighbor_indices, :
                    ]
                    distance = np.linalg.norm(
                        child_position - neighbor_position_mat, axis=1
                    )
                else:
                    distance = np.array([])

                # boolean array specifying if there is no overlap between
                # the proposed child position and the other cells
                no_overlap = np.all(distance >= 2 * self.cell_radius)
                # if it is true that there is no overlap for
                # every element of the array, we break the loop
                if no_overlap:
                    break

            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available
            # space
            if no_overlap:
                # we create a child in that position
                if cell.is_stem:
                    random_number = self.rng.random()
                    if random_number <= self.prob_stem:  # ps
                        child_cell = CellLite(
                            position=child_position,
                            culture=self,
                            is_stem=True,
                            parent_index=cell_index,
                        )
                    else:
                        child_cell = CellLite(
                            position=child_position,
                            culture=self,
                            is_stem=False,
                            parent_index=cell_index,
                        )
                        if random_number <= (
                            self.prob_stem + self.prob_diff
                        ):  # pd
                            cell.is_stem = False
                        elif (
                            self.rng.random() <= self._swap_probability
                        ):  # pa = 1-ps-pd
                            cell.is_stem = False
                            child_cell.is_stem = True
                else:
                    child_cell = CellLite(
                        position=child_position,
                        culture=self,
                        is_stem=False,
                        parent_index=cell_index,
                    )
                # # we add this cell to the culture's cells and active_cells lists
                # self.cells.append(child_cell)
                # self.active_cells.append(child_cell)
                # # we add the parent as first neighbor (necessary for
                # # the find_neighbors that are not from_entire_culture)
                # # First, we calculate the index of the child
                # child_index = len( # esto tiene pinta que ya se calculó en el init de CellLite ########################################################
                #     self.cells
                # ) - 1  # cause the index of the cell *is* the length of the list minus 1
                # # we add the parent as neighbor of the child
                child_cell.neighbors_indexes.add(cell_index)
                # we find the child's neighbors
                self.find_neighbors(child_cell._position_index)
                # we add the child as a neighbor of its neighbors
                for neighbor_index in self.cells[
                    child_cell._position_index
                ].neighbors_indexes:
                    self.cells[neighbor_index].neighbors_indexes.add(
                        child_cell._position_index
                    )
            else:
                cell.available_space = False
                self.active_cells.remove(cell._position_index)
                # if there was no available space, we turn off reproduction
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing
        # (reproduction is turned off)

    # ---------------------------------------------------------

    def simulate_with_persistent_data(
        self, num_times: int, culture_name: str
    ) -> None:
        """Simulate culture growth for a specified number of time steps and
        record the data in a file at each time step, so its available in real
        time.

        At each time step, we randomly sort the list of active cells and then
        we tell them to reproduce one by one. The data that gets recorded at
        each step is the total number of cells, the number of active cells,
        the number of stem cells, and the number of active stem cells. It no
        longer saves data in a dictionary, but rather to a file with a
        specified name.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.
        culture_name : str
            The name of the culture in the simulation, in the format
            culture_pd={prob_diff}_ps={prob_stem}_realization_{j}.dat
        """

        # we count the initial amount of CSCs
        if self.first_cell_is_stem:
            initial_amount_of_csc = 1
        else:
            initial_amount_of_csc = 0

        # we write the header and the data values for this time step
        with open(f"data/{culture_name}.dat", "w") as file:
            file.write("total, active, total_stem, active_stem \n")
            file.write(
                f"1, 1, {initial_amount_of_csc}, {initial_amount_of_csc} \n"
            )

        # we simulate for num_times time steps
        for i in range(1, num_times):
            # we get a permuted copy of the cells list
            active_cell_indexes = self.rng.permutation(self.active_cells)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for index in active_cell_indexes:
                self.reproduce(index)

            # we count the number of CSCs in this time step
            total_stem_counter = 0
            for cell in self.cells:
                if cell.is_stem:
                    total_stem_counter = total_stem_counter + 1

            # we count the number of active CSCs in this time step
            active_stem_counter = 0
            for index in self.active_cells:
                if self.cells[index].is_stem:
                    active_stem_counter = active_stem_counter + 1

            # we save the data to a file
            with open(f"data/{culture_name}.dat", "a") as file:
                file.write(
                    f"{len(self.cells)}, {len(self.active_cells)}, {total_stem_counter}, {active_stem_counter} \n"
                )
