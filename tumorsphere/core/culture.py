"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""

from datetime import datetime
from typing import Set

import numpy as np
from collections import defaultdict

from tumorsphere.core.cells import Cell
from tumorsphere.core.output import TumorsphereOutput
from tumorsphere.core.forces import Force


class Culture:
    def __init__(
        self,
        output: TumorsphereOutput,
        force: Force,
        adjacency_threshold: float = 4,
        cell_radius: float = 1,
        cell_max_repro_attempts: int = 1000,
        cell_max_def_attempts: int = 10,
        first_cell_is_stem: bool = True,
        prob_stem: float = 0,
        prob_diff: float = 0,
        rng_seed: int = 110293658491283598,
        swap_probability: float = 0.5,
        number_of_cells: int = 5,
        density: float = 0.5,
        reproduction: bool = False,
        movement: bool = True,
        cell_area: float = np.pi,
        stabilization_time: int = 200,
        max_time_deformation_initial: int = 500,
        max_time_deformation: int = 2500,
        threshold_overlap: float = 0.61,
        delta_t: float = 0.01,
        aspect_ratio_max: float = 5,
    ):
        """
        Initialize a new culture of cells.

        Parameters
        ----------
        adjacency_threshold : int, optional
            The maximum distance at which two cells can be considered
            neighbors, by default 4.
        cell_radius : int, optional
            The radius of a cell, by default 1.
        cell_max_repro_attempts : int, optional
            The maximum number of reproduction attempts a cell can make,
            by default 1000.
        cell_max_def_attempts : int, optional
            The maximum number of deformation attempts a cell can make,
            by default 10.
        first_cell_is_stem : bool, optional
            Whether the first cell is a stem cell or not, by default False.
        prob_stem : float, optional
            The probability that a cell becomes a stem cell, by default 0.
        prob_diff : float, optional
            The probability that a cell differentiates, by default 0.
        rng_seed : int, optional
            Seed for the random number generator, by default
            110293658491283598.
        number_of_cells : int, optional
            The number of cells in the culture.
        density : float
            The density of the cells in the culture. Used to calculate the side 
            length of the box.
        reproduction : bool
            Whether the cells reproduces or not.
        movement : bool
            Whether the cells moves or not.
        cell_area : float
            the area of all cells in the culture.
        stabilization_time : int
            the time we have to wait in order to start the deformation
        threshold_overlap : float
            the threshold for the overlap for which the cells start to interact and deform
        delta_t : float
            the time interval used to move
        apect_ratio_max : float
            the max value of the aspect ratio that a cell can have after deforms

        Attributes
        ----------
        cell_max_repro_attempts : int
            Maximum number of reproduction attempts a cell can make.
        cell_max_def_attempts : int
            Maximum number of deformation attempts a cell can make.
        adjacency_threshold : int
            The maximum distance at which two cells can be considered
            neighbors.
        cell_radius : int
            The radius of a cell.
        prob_stem : float
            The probability that a cell becomes a stem cell.
        prob_diff : float
            The probability that a cell differentiates.
        swap_probability : float
            The probability that a cell swaps its type with its offspring.
        number_of_cells : int, optional
            The number of cells in the culture.
        side : int, optional
            The length of the side of the square where the cells move.
        reproduction : bool
            Whether the cells reproduce or not
        movement : bool
            Whether the cells move or not
        cell_area : float
            the area of all cells in the culture.
        stabilization_time : int
            the time we have to wait in order to start the deformation
        threshold_overlap : float
            the threshold for the overlap for which the cells start to interact and deform
            to deform
        delta_t : float
            the time interval used to move
        apect_ratio_max : float
            the max value of the aspect ratio that a cell can have after deforms
        rng : numpy.random.Generator
            Random number generator.
        first_cell_is_stem : bool
            Whether the first cell is a stem cell or not.
        cell_positions : numpy.ndarray
            Matrix to store the positions of all cells in the culture.
        cell_velocities : numpy.ndarray
            Matrix to store the velocities of all cells in the culture.
        cells : list[Cell]
            List of all cells in the culture.
        active_cells : list[Cell]
            List of all active cells in the culture.
        """

        # cell attributes
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.cell_max_def_attempts = cell_max_def_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.swap_probability = swap_probability
        self.number_of_cells = number_of_cells
        self.reproduction = reproduction
        self.movement = movement
        self.cell_area = cell_area
        self.threshold_overlap = threshold_overlap
        self.delta_t = delta_t
        self.aspect_ratio_max = aspect_ratio_max
        self.density = density

        # calculation of the side using other parameters
        self.side = np.sqrt(self.number_of_cells * self.cell_area / self.density)

        # we instantiate the culture's RNG with the provided entropy
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # initialize the positions matrix
        self.cell_positions = np.empty((0, 3), float)

        # and the velocities matrix
        self.cell_phies = np.array([])

        # we initialize the lists of cells
        self.cells = []
        self.active_cell_indexes = []

        # time at wich the culture was created
        self.simulation_start = self._get_simulation_time()

        self.output = output

        self.force = force

        # time that passes until the system stabilizes from the initial condition
        self.stabilization_time = stabilization_time
        self.max_time_deformation = max_time_deformation
        self.max_time_deformation_initial = max_time_deformation_initial

    # ----------------database related behavior----------------

    def _get_simulation_time(self):
        # we get the current date and time
        current_time = datetime.now()
        # we format the string
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return time_string

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
                new_neighbors_l2: Set[int] = (
                    cell2.neighbors_indexes.difference(
                        neighbors_up_to_third_degree
                    )
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

        This method updates the cell's list of indexes of neighbors in-place.
        When a new neighbor is found, the cell is also added to the list of
        neighbors of this newly found neighbor.

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
        candidate_set = neighbors_up_to_certain_degree - cell.neighbors_indexes
        candidate_set.difference_update([cell_index])
        candidate_indexes = list(candidate_set)

        if len(candidate_indexes) > 0:
            candidate_positions = self.cell_positions[candidate_indexes, :]
            candidate_distances = np.linalg.norm(
                self.cell_positions[cell_index] - candidate_positions, axis=1
            )

            for candidate_index, candidate_distance in zip(
                candidate_indexes, candidate_distances
            ):
                in_neighborhood = (
                    candidate_distance <= self.adjacency_threshold
                )
                if in_neighborhood:
                    cell.neighbors_indexes.add(candidate_index)
                    self.cells[candidate_index].neighbors_indexes.add(
                        cell_index
                    )

    def generate_new_position(self, cell_index: int) -> np.ndarray:
        """Generate a proposed position for the child cell, adjacent to the
        given one.

        A new position for the child cell is randomly generated, at a distance
        equals to two times the radius of a cell (all cells are assumed to
        have the same radius). This is done by randomly choosing the angular
        spherical coordinates from a uniform distribution. It uses the cell
        current position and its radius.

        Returns
        -------
        new_position : numpy.ndarray
            A 3D vector representing the new position of the cell.

        Notes
        -----
        - All cells are assumed to have the same radius.
        - To get a uniform distribution of points in the unit sphere, we have
        to choose cos(theta) uniformly in [-1, 1] instead of theta uniformly
        in [0, pi].
        """
        cos_theta = self.rng.uniform(low=-1, high=1)
        theta = np.arccos(cos_theta)  # Convert cos(theta) to theta
        phi = self.rng.uniform(low=0, high=2 * np.pi)

        x = 2 * self.cell_radius * np.sin(theta) * np.cos(phi)
        y = 2 * self.cell_radius * np.sin(theta) * np.sin(phi)
        z = 2 * self.cell_radius * np.cos(theta)
        cell_position = self.cell_positions[cell_index]
        new_position = cell_position + np.array([x, y, z])
        return new_position

    def reproduce(self, cell_index: int, tic: int) -> None:
        """The given cell reproduces, generating a new child cell.

        Attempts to create a new cell in a random position, adjacent to the
        current cell, if the cell has available space. If the cell fails to
        find a position that doesn't overlap with existing cells, (for the
        estabished maximum number of attempts), no new cell is created, and
        the current one is deactivated. This means that we set its available
        space to `False` and remove it from the list of active cells.

        Notes
        -----
        The `if cell.available_space` might be redundant since we remove the
        cells from the `active_cells` list when seting that to `False`, but
        the statement is kept as a way of double checking.
        """

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
                        child_cell = Cell(
                            position=child_position,
                            culture=self,
                            is_stem=True,
                            parent_index=cell_index,
                            creation_time=tic,
                        )
                    else:
                        child_cell = Cell(
                            position=child_position,
                            culture=self,
                            is_stem=False,
                            parent_index=cell_index,
                            creation_time=tic,
                        )
                        if random_number <= (
                            self.prob_stem + self.prob_diff
                        ):  # pd
                            cell.is_stem = False
                            self.output.record_stemness(
                                cell_index, tic, cell.is_stem
                            )
                        elif (
                            self.rng.random() <= self.swap_probability
                        ):  # pa = 1-ps-pd
                            cell.is_stem = False
                            self.output.record_stemness(
                                cell_index, tic, cell.is_stem
                            )
                            child_cell.is_stem = True
                            self.output.record_stemness(
                                child_cell._index, tic, child_cell.is_stem
                            )
                else:
                    child_cell = Cell(
                        position=child_position,
                        culture=self,
                        is_stem=False,
                        parent_index=cell_index,
                        creation_time=tic,
                    )
                # we add the parent as neighbor of the child
                child_cell.neighbors_indexes.add(cell_index)
                # we find the child's neighbors
                self.find_neighbors(child_cell._index)
                # we add the child as a neighbor of its neighbors
                for neighbor_index in self.cells[
                    child_cell._index
                ].neighbors_indexes:
                    self.cells[neighbor_index].neighbors_indexes.add(
                        child_cell._index
                    )
            else:
                # The cell has no available space to reproduce
                cell.available_space = False
                # We no longer consider it active, so we remove *all* of its
                # instances from the list of active cell indexes
                set_of_current_active_cells = set(self.active_cell_indexes)
                set_of_current_active_cells.discard(cell_index)
                self.active_cell_indexes = list(set_of_current_active_cells)
                # We record the deactivation
                self.output.record_deactivation(cell_index, tic)
                # if there was no available space, we turn off reproduction
                # and record the change in the Cells table of the DataBase
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing
        # (reproduction is turned off)

    # ---------------------------------------------------------

    def relative_pos(self, cell_position: float, neighbor_position: float):
        """
        It calculates the relative position in x and y of 2 cells taking into account
        that they move in a box.

        Parameters
        ----------
        cell_position : float
            The position of the cell.
        neighbor_position : int
            The position of the neighbor.

        Returns
        -------
        relative_pos_x : float
            The x component of the relative position of the cells.
        relative_pos_y : float
            The y component of the relative position of the cells.
        """

        relative_pos_x = -(neighbor_position[0] - cell_position[0])
        relative_pos_y = -(neighbor_position[1] - cell_position[1])
        abs_rx = abs(relative_pos_x)
        abs_ry = abs(relative_pos_y)

        # we choose the distance between two cells as the shortest distance taking into account the box
        if abs_rx > 0.5 * self.side:
            relative_pos_x = np.sign(relative_pos_x) * (abs_rx - self.side)
        if abs_ry > 0.5 * self.side:
            relative_pos_y = np.sign(relative_pos_y) * (abs_ry - self.side)

        return relative_pos_x, relative_pos_y

    # ---------------------------------------------------------
    def calculate_overlap(
        self,
        cell_index: int,
        neighbor_index: int,
        relative_pos_x: float,
        relative_pos_y: float,
    ):
        """
        It calculates the overlap between two cells, allowing evaluation of a new
        orientation (phi) and aspect ratio for the cell.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        neighbor_index : int
            The index of the neighbor.
        relative_pos_x : float
            The x component of the relative position of the cells.
        relative_pos_y : float
            The y component of the relative position of the cells.
        Returns
        -------
        overlap : float
            The overlap between cells
        """
        cell = self.cells[cell_index]
        neighbor = self.cells[neighbor_index]

        # we introduce the anisotropy (eps) and the diagonal squared (alpha) of the cell
        eps_cell = (cell.aspect_ratio**2 - 1) / (cell.aspect_ratio**2 + 1)
        # alpha = l_parallel**2+l_perp**2
        # with l_parallel = np.sqrt((cell_area*cell.aspect_ratio)/np.pi)
        # and l_perp = sqrt(cell_area/(np.pi*cell.aspect_ratio))
        alpha_cell = (self.cell_area / np.pi) * (
            cell.aspect_ratio + 1 / cell.aspect_ratio
        )

        # and the neighbor
        eps_neighbor = (neighbor.aspect_ratio**2 - 1) / (
            neighbor.aspect_ratio**2 + 1
        )
        alpha_neighbor = (self.cell_area / np.pi) * (
            neighbor.aspect_ratio + 1 / neighbor.aspect_ratio
        )

        # now we introduce the constant beta introduced by us in the TF
        beta = (
            (alpha_cell + alpha_neighbor) ** 2
            - (alpha_cell * eps_cell - alpha_neighbor * eps_neighbor) ** 2
            - 4
            * alpha_cell
            * eps_cell
            * alpha_neighbor
            * eps_neighbor
            * (
                np.cos(
                    self.cell_phies[cell_index]
                    - self.cell_phies[neighbor_index]
                )
            )
            ** 2
        )

        # then we calculate the nematic matrixes/tensor
        Q_cell = np.array(
            [
                [
                    np.cos(2 * self.cell_phies[cell_index]),
                    np.sin(2 * self.cell_phies[cell_index]),
                    0,
                ],
                [
                    np.sin(2 * self.cell_phies[cell_index]),
                    -np.cos(2 * self.cell_phies[cell_index]),
                    0,
                ],
                [0, 0, 0],
            ]
        )

        Q_neighbor = np.array(
            [
                [
                    np.cos(2 * self.cell_phies[neighbor_index]),
                    np.sin(2 * self.cell_phies[neighbor_index]),
                    0,
                ],
                [
                    np.sin(2 * self.cell_phies[neighbor_index]),
                    -np.cos(2 * self.cell_phies[neighbor_index]),
                    0,
                ],
                [0, 0, 0],
            ]
        )

        # and calculate the matriz M

        matrix_M = (
            alpha_cell * eps_cell * Q_cell
            + alpha_neighbor * eps_neighbor * Q_neighbor
        ) / (alpha_cell + alpha_neighbor)

        # finally we can calculate i_0 and the overlap
        # i_0 = (4*pi*l_par_k*l_perp_k*l_par_j*l_perp_j)/sqrt(beta)
        # with l_parallel = np.sqrt((cell_area*cell.aspect_ratio)/np.pi)
        # and l_perp = sqrt(cell_area/(np.pi*cell.aspect_ratio))
        i_0 = 4 * self.cell_area**2 / (np.pi * np.sqrt(beta))

        relative_pos = np.array([relative_pos_x, relative_pos_y, 0])
        overlap = i_0 * np.exp(
            -((alpha_cell + alpha_neighbor) / beta)
            * np.matmul(
                relative_pos,
                np.matmul(np.identity(3) - matrix_M, relative_pos),
            )
        )
        # we return the overlap between the cell and its neighbor
        return overlap


    # ---------------------------------------------------------
    def assign_to_grid(self):
        """
        Assigns each cell to a box in a 2D grid. The side of each box is approx 2 times 
        the semimajor axis of a cell with phi=5, and it has to be so that the number of 
        boxes is an integer.

        Returns
        -------
        grid : dict
            The grid with all the boxes and the cells inside them.
        r_0 : float
            The length of the side of the boxes.

        """
        # Create an empty grid as a dictionary
        grid = defaultdict(list)
        # the number of cells in the grid (called boxes to diferenciate from cells)
        # is choose in order to have side of the box equal to 2*semi_major_axis
        # for the largest cell
        num_boxes = int(np.floor(self.side / (2*np.sqrt((self.cell_area * 5) / np.pi))))
        r_0 = self.side/num_boxes
        for i, pos in enumerate(self.cell_positions):
            # Turn the position into an index
            index = tuple((pos // r_0).astype(int))
            # Add the index to the correspond box
            grid[index].append(i)
        return grid, r_0
    
    # ---------------------------------------------------------
    def find_neighbors_in_grid(self, grid, r_0):
        """
        It searchs for the neighbor candidates of every cell. It is a candidate for 
        neighbor if the cell is in a neighbor box.

        Parameters
        ----------
        grid : dict
            The grid with all the boxes and the cells inside them.
        r_0 : float
            The length of the side of the boxes.

        Returns
        -------
        neighbors : dict
            A dict with all the neighbors of every cell.
        """
        # Create a dict to save the neighbors
        neighbors = defaultdict(set)
        num_boxes = int(np.floor(self.side / (2*np.sqrt((self.cell_area * 5) / np.pi))))
        for box_index, box_cells in grid.items():
            # Loop for neighboring boxes including the current
            step = [-1, 0, 1]
            for dx in step:
                for dy in step:
                    # we see the neighbors taking into account periodic conditions
                    neighbor_box = (np.mod(box_index[0] + dx, num_boxes), np.mod(box_index[1] + dy, num_boxes), 0)
                    if neighbor_box in grid:
                        # Compare cells in the current box with the cells in neighboring boxes
                        for i in box_cells:
                            for j in grid[neighbor_box]:
                                if i != j:
                                        neighbors[i].add(j)
                                        neighbors[j].add(i)
        return neighbors
    
    # ---------------------------------------------------------
    def interaction(self, cell_index: int, delta_t: float):
        """The given cell interacts with others if they are close enough.

        It describes the interaction of the cells given a force. It changes the position
        of the cell (because of the forces exerted by all the other cells and the
        intrinsic velocity) and it's angle in the x-y plane, phi (becuase of a torque).

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        delta_t : float
            The time step.

        Returns
        -------
        dif_position : np.ndarray
            The change in the position of the cell.
        dphi : np.ndarray
            The change in the angle phi of the cell.
        -----
        """
        cell = self.cells[cell_index]

        # We get the neighbors of the cell
        candidate_neighbors = cell.neighbors_indexes.copy()

        # Calculate relative positions for all neighbors
        relative_positions = np.array(
            [
                self.relative_pos(
                    self.cell_positions[cell_index],
                    self.cell_positions[neighbor_index],
                )
                for neighbor_index in candidate_neighbors
            ]
        )
        # Filter neighbors whose distance is less than the sum of the major semi axes
        filtered_neighbors = [
            (neighbor_index, relative_pos)
            for neighbor_index, relative_pos in zip(candidate_neighbors, relative_positions)
            if np.linalg.norm(relative_pos) < (
                np.sqrt((self.cell_area * cell.aspect_ratio) / np.pi)
                + np.sqrt((self.cell_area * self.cells[neighbor_index].aspect_ratio) / np.pi)
            )
        ]
        # Calculate overlap for filtered neighbors
        overlaps = [
            self.calculate_overlap(cell_index, neighbor_index, relative_pos[0], relative_pos[1])
            for neighbor_index, relative_pos in filtered_neighbors
        ]
        # The significant neighbors are those which the overlap is less than the threshold
        significant_neighbors = [
            neighbor_index
            for (neighbor_index, relative_pos), overlap in zip(filtered_neighbors, overlaps)
            if overlap > self.threshold_overlap
        ]
        # same with positions
        significant_relative_positions = [
            relative_pos
            for (neighbor_index, relative_pos), overlap in zip(filtered_neighbors, overlaps)
            if overlap > self.threshold_overlap
        ]
        # Calculate interaction with significant neighbors
        dif_position, dif_phi = self.force.calculate_interaction(
            self.cells,
            self.cell_phies,
            cell_index,
            [(neigh, rel_pos) for (neigh, rel_pos) in zip(significant_neighbors, significant_relative_positions)],
            delta_t,
            self.cell_area,
        )
        # Create a set with the indexes that are in candidate_neighbors but not in significant_neighbors
        significant_neighbors_set = set(significant_neighbors)
        failed_candidates = candidate_neighbors - significant_neighbors_set
        # For those indexes we eliminate the current cell of their neighbors
        for failed_index in failed_candidates:
            if cell_index in self.cells[failed_index].neighbors_indexes:
                self.cells[failed_index].neighbors_indexes.remove(cell_index)
        #we return the change in the position and in the phi angle of the cell
        return dif_position, dif_phi

    # ---------------------------------------------------------
    def move(
        self,
        dif_positions: np.ndarray,
        dif_phies: np.ndarray,
    ) -> None:
        """The given cell moves with a given velocity and changes its orientation.

        Attempts to move one step with a particular velocity and changes its orientation.
        If the cell arrives to a border of the culture's square, it appears on the other
        side.

        Parameters
        ----------
        dif_positions : np.ndarray
            Matrix that contains the changes in position of all the cells.
        dif_phies : np.ndarray
            Matrix that contains the changes in orientation of all the cells.
        -----
        """
        # Updating the cell's position
        self.cell_positions = self.cell_positions + dif_positions

        # and the angle
        self.cell_phies = self.cell_phies + dif_phies

        # Enforcing boundary condition
        self.cell_positions = np.mod(self.cell_positions, self.side)

    # ---------------------------------------------------------
    def generate_new_position_2D(
        self, cell_index: int, new_phi: float, new_aspect_ratio: float
    ):
        """Generate a proposed position for the cell, given a new phi and a new aspect
        ratio that help us to know if there is space available to deform the
        cell.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        new_phi : float
            The orientation of the new cell.
        new_aspect_ratio : float
            The aspect ratio of the new cell.

        Returns
        -------
        new_position : numpy.ndarray
            A 3D vector representing the new position of the cell.
        """

        new_semi_major_axis = np.sqrt(
            (self.cell_area * new_aspect_ratio) / np.pi
        )
        old_semi_major_axis = np.sqrt(
            (self.cell_area * self.cells[cell_index].aspect_ratio) / np.pi
        )
        old_semi_minor_axis = np.sqrt(
            self.cell_area / (np.pi * self.cells[cell_index].aspect_ratio)
        )

        d = np.sqrt(
            old_semi_major_axis**2
            * (np.cos(new_phi - self.cell_phies[cell_index])) ** 2
            + old_semi_minor_axis**2
            * (np.sin(new_phi - self.cell_phies[cell_index])) ** 2
        )
        x = (new_semi_major_axis - d) * np.cos(new_phi)
        y = (new_semi_major_axis - d) * np.sin(new_phi)

        new_position = self.cell_positions[cell_index] + np.array([x, y, 0])
        new_position = np.mod(new_position, self.side)
        return new_position

    # ---------------------------------------------------------
    def deformation(self, cell_index: int, candidate_neighbors_total, grid, r_0):
        """If the cell is round, an angle is chosen randomly.
        If the new cell with these angle and aspect ratio = maximum (given as an
        attribute) does not overlap with others, it remains.
        If not, try again up to cell_max_def_attempts.
        If it fails to deform, it remains as it was originally.

        Parameters
        ----------
        cell_index : int
            The index of the cell.

        Return
        ----------
        succesful_deformation : bool
            True if the deformation was successful, False otherwise.
        """
        cell = self.cells[cell_index]
        if np.isclose(cell.aspect_ratio, 1):
            # we save the old attributes
            old_position = np.array(self.cell_positions[cell_index])
            old_phi = self.cell_phies[cell_index]
            old_aspect_ratio = cell.aspect_ratio

            for attempt in range(self.cell_max_def_attempts):
                # random phi and aspect ratio=max and generate a position with them
                new_phi = self.rng.uniform(low=0, high=2 * np.pi)
                new_aspect_ratio = self.aspect_ratio_max
                new_position = self.generate_new_position_2D(
                    cell_index, new_phi, new_aspect_ratio
                )
                # updating attributes
                self.cell_positions[cell_index] = new_position
                self.cell_phies[cell_index] = new_phi
                self.cells[cell_index].aspect_ratio = new_aspect_ratio

                # We get the neighbors of the cell
                # We see if the cell correspond to the same tupple in the grid as before
                #num_boxes = int(np.ceil(self.side / r_0))
                old_box_index = tuple((old_position // r_0).astype(int))
                #old_box_index = tuple(np.mod(old_position // r_0, num_boxes).astype(int))
                #new_box_index = tuple(np.mod(new_position // r_0, num_boxes).astype(int))
                new_box_index = tuple((new_position // r_0).astype(int))
                # initialization candidate_neighbors
                candidate_neighbors = set()
                # if they are equal, then the neighbors are the same as before:
                if old_box_index == new_box_index:
                    candidate_neighbors = candidate_neighbors_total[cell_index].copy()
                # if they are different, we have to calculate the new neighbors
                else:
                    # we see the cells that are in the box of the correspond new position
                    cells_same_box = set(grid.get(new_box_index, []))
                    if cells_same_box:
                        # Get the neighbors of a random cell in the same box
                        reference_cell = next(iter(cells_same_box))
                        #  and remove the current cell
                        candidate_neighbors = candidate_neighbors_total[reference_cell].copy() - {cell_index}
                        # Add the reference cell as a neighbor
                        candidate_neighbors.add(reference_cell)
                    else:
                        # we calculate the neighbors again
                        num_boxes = int(np.floor(self.side / (2*np.sqrt((self.cell_area * 5) / np.pi))))
                        step = [-1, 0, 1]
                        # watch for all the neighbor boxes
                        for dx in step:
                            for dy in step:
                                # we see the neighbors taking into account periodic conditions
                                neighbor_box = (np.mod(new_box_index[0] + dx, num_boxes), np.mod(new_box_index[1] + dy, num_boxes), 0)
                                if neighbor_box in grid:
                                    # if the cells are in neighbor boxes, then they are candidate for neighbors
                                        for j in grid[neighbor_box]:
                                            if cell_index != j:
                                                    candidate_neighbors.add(j)
                # Calculate relative positions for all neighbors
                relative_positions = np.array(
                    [
                        self.relative_pos(
                            self.cell_positions[cell_index],
                            self.cell_positions[neighbor_index],
                        )
                        for neighbor_index in candidate_neighbors
                    ]
                )
                # Filter neighbors whose distance is less than the sum of the major semi axes
                filtered_neighbors = [
                    (neighbor_index, relative_pos)
                    for neighbor_index, relative_pos in zip(candidate_neighbors, relative_positions)
                    if np.linalg.norm(relative_pos) < (
                        np.sqrt((self.cell_area * cell.aspect_ratio) / np.pi)
                        + np.sqrt((self.cell_area * self.cells[neighbor_index].aspect_ratio) / np.pi)
                    )
                ]
                
                # calculation of overlap
                no_overlap = True
                for neighbor_index, relative_pos in filtered_neighbors:
                    relative_pos_x, relative_pos_y= relative_pos[0], relative_pos[1]

                    overlap = self.calculate_overlap(
                        cell_index,
                        neighbor_index,
                        relative_pos_x,
                        relative_pos_y,
                    )
                    if overlap > self.threshold_overlap:
                        # if the new cell overlaps with another, we turn back to the
                        # original values
                        self.cell_positions[cell_index] = old_position
                        self.cell_phies[cell_index] = old_phi
                        self.cells[cell_index].aspect_ratio = old_aspect_ratio
                        no_overlap = False
                        break
                if no_overlap:
                    # if there is no overlap, the new cell remains and we finish the loop
                    succesful_deformation = True
                    # we add the cell as a neighbor of its new neighbors
                    if old_box_index != new_box_index:
                        for neighbor_index in candidate_neighbors:
                            candidate_neighbors_total[neighbor_index].add(cell_index)
                        # and we eliminate it of the old neighbors
                        ex_neighbors=candidate_neighbors_total[cell_index]-candidate_neighbors
                        for neighbor_index in ex_neighbors:
                            candidate_neighbors_total[neighbor_index].discard(cell_index)
                    return succesful_deformation
            succesful_deformation = False
            return succesful_deformation
        else:
            succesful_deformation = False
            return succesful_deformation

    # ---------------------------------------------------------

    def simulate(self, num_times: int) -> None:
        """Simulate culture growth/movement for a specified number of time steps.

        For reproduction, at each time step, we randomly sort the list of active cells
        and thenwe tell them to reproduce one by one.

        For movement at each time step we move the cells.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.
        """

        # if the culture is brand-new, we create the tables of the DB and the
        # first cell
        if len(self.cells) == 0:
            # we insert the register corresponding to this culture
            self.output.begin_culture(
                self.prob_stem,
                self.prob_diff,
                self.rng_seed,
                self.simulation_start,
                self.adjacency_threshold,
                self.swap_probability,
            )

            # we instantiate the first cell (only if reproduction)
            if self.reproduction:
                first_cell_object = Cell(
                    position=np.array([0, 0, 0]),
                    culture=self,
                    is_stem=self.first_cell_is_stem,
                    parent_index=0,
                    available_space=True,
                )
            else:
                pass

            # We add all the cells in the case of movement
            if self.movement:
                for i in range(0, self.number_of_cells):
                    # choose a random position and angle in the xy plane (phi)
                    Cell(
                        position=np.array(
                            [
                                self.rng.uniform(low=0, high=self.side),
                                self.rng.uniform(low=0, high=self.side),
                                0,
                            ]
                        ),
                        culture=self,
                        is_stem=self.first_cell_is_stem,
                        phi=0,
                        aspect_ratio=1,
                        parent_index=0,
                        available_space=True,
                    )
        # Save the data (for dat, ovito, and/or SQLite)
        self.output.record_culture_state(
            tic=0,
            cells=self.cells,
            cell_positions=self.cell_positions,
            cell_phies=self.cell_phies,
            active_cell_indexes=self.active_cell_indexes,
            side=self.side,
            cell_area=self.cell_area,
        )
        # we simulate for num_times time steps
        reproduction = self.reproduction
        movement = self.movement

        # we initialiaze the last deformation as the first time when they attempt to deform
        last_time_deformation = self.stabilization_time
        for i in range(1, num_times + 1):
            # and reproduce or move the cells in this random order
            if reproduction:
                # we get a permuted copy of the cells list
                active_cell_indexes = self.rng.permutation(
                    self.active_cell_indexes
                )
                for index in active_cell_indexes:
                    self.reproduce(cell_index=index, tic=i)

            if movement:
                # We assign each cell to a box in the grid
                grid, r_0 = self.assign_to_grid()
                # We find the candidates for neighbors
                candidate_neighbors_total = self.find_neighbors_in_grid(grid, r_0)
                dif_positions = np.zeros((len(self.active_cell_indexes), 3))
                dif_phies = np.zeros(len(self.active_cell_indexes))
                for index in self.active_cell_indexes:
                    cell = self.cells[index]
                    # we update the neighbors as the candidates
                    cell.neighbors_indexes = candidate_neighbors_total[index].copy()
                for index in self.active_cell_indexes:
                    dif_position, dif_phi = self.interaction(
                        cell_index=index, delta_t=self.delta_t
                    )
                    dif_positions[index] = dif_position
                    dif_phies[index] = dif_phi

                self.move(dif_positions=dif_positions, dif_phies=dif_phies)

                # If the time step is less than the stabilization time, cells dont deform 
                if i <= self.stabilization_time:
                    deform = False
                # If the time step is between the stabilization time and the max, cells deform
                elif self.stabilization_time<i<self.max_time_deformation_initial:
                    deform = True
                # If we are in the max_time, then we see if the cells are all round and in that case,
                # they stop deforming. (This value of deform stays for later steps)
                elif i == self.max_time_deformation_initial:
                    aspect_ratios = np.array([self.cells[i].aspect_ratio for i in self.active_cell_indexes])
                    # Set deform to False if all aspect ratios are 1, otherwise True
                    deform = not np.all(aspect_ratios == 1)
                # Finally, if the time is larger than the sum between the last time and
                # the max_time. (This value of deform stays for later steps)
                elif i > last_time_deformation + self.max_time_deformation:
                    deform = False
                # In order to be sure that we elongate all the cells possible, we add that 
                # in the last steps the cells try to deform again.
                #elif i > num_times-self.stabilization_time:
                #    deform = True
                # Only deform if deform is still True
                if deform:
                    # we deform using the candidates for neighbors
                    succesful_deformations = [
                        self.deformation(cell_index=index, candidate_neighbors_total=candidate_neighbors_total, grid=grid, r_0=r_0) for index in self.active_cell_indexes
                    ]
                    if any(succesful_deformations):
                        last_time_deformation = i
                    

            # Save the data (for dat, ovito, and/or SQLite)
            self.output.record_culture_state(
                tic=i,
                cells=self.cells,
                cell_positions=self.cell_positions,
                cell_phies=self.cell_phies,
                active_cell_indexes=self.active_cell_indexes,
                side=self.side,
                cell_area=self.cell_area,
            )
