"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""

from datetime import datetime
from typing import Set

import numpy as np

from tumorsphere.core.cells import Cell
from tumorsphere.core.output import TumorsphereOutput


class Culture:
    def __init__(
        self,
        output: TumorsphereOutput,
        adjacency_threshold: float = 4,
        cell_radius: float = 1,
        cell_max_repro_attempts: int = 1000,
        first_cell_is_stem: bool = True,
        prob_stem: float = 0,
        prob_diff: float = 0,
        rng_seed: int = 110293658491283598,
        swap_probability: float = 0.5,
        number_of_cells: int = 5,
        side: int = 10,
        reproduction: bool = False,
        movement: bool = True,
        kProp: float = 1,
        kRep: float = 10,
        bExp: float = 3,
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
        side : int, optional
            The length of the side of the square where the cells move.
        reproduction : bool
            Whether the cells reproduces or not
        movement : bool
            Whether the cells moves or not
        kProp : float
            #
        kRep : float
            #
        bExp : float
            #
        
        Attributes
        ----------
        cell_max_repro_attempts : int
            Maximum number of reproduction attempts a cell can make.
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
        kProp : float
            #
        kRep : float
            #
        bExp : float
            #
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
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.swap_probability = swap_probability
        self.number_of_cells = number_of_cells
        self.side = side
        self.reproduction = reproduction
        self.movement = movement

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

        # interaction parameters
        self.kProp = kProp
        self.kRep = kRep
        self.bExp = bExp

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
    def relative_pos(self, cell_index: int, neighbor_index: int):
        """
        It calculates the relative position in x and y of 2 cells taking into account
        that they move in a box.

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        neighbor_index : int
            The index of the neighbor.

        Returns
        -------
        relative_pos_x : float
            The x component of the relative position of the cells.
        relative_pos_y : float
            The y component of the relative position of the cells.
        """

        relative_pos_x = -(
            self.cell_positions[neighbor_index][0]
            - self.cell_positions[cell_index][0]
        )
        relative_pos_y = -(
            self.cell_positions[neighbor_index][1]
            - self.cell_positions[cell_index][1]
        )
        abs_rx = abs(relative_pos_x)
        abs_ry = abs(relative_pos_y)

        # we choose the distance between two cells as the shortest distance taking into account the box
        if abs_rx > 0.5 * self.side:
            relative_pos_x = np.sign(relative_pos_x) * (abs_rx - self.side)
        if abs_ry > 0.5 * self.side:
            relative_pos_y = np.sign(relative_pos_y) * (abs_ry - self.side)

        return relative_pos_x, relative_pos_y

    # ---------------------------------------------------------
    def interaction_between_2_cells(
        self,
        cell_index: int,
        neighbor_index: int,
        alpha: float,
        distance_sq: float,
        s_epsA05: float,
        s_epsA2: float,
        s_diag2: float,
    ):
        """
        It describes the interaction between 2 elongated cells using the model developed
        in [Grossman2020].

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        neighbor_index : int
            The index of the neighbor.
        alpha : float
            The angle of the relative vector position with the x axis.
        distance_sq : float
            The square of the relative distance between the cells.
        s_epsA05 : float
            The half of the anisotropy of the cell.
        s_epsA2 : float
            The square of the anisotropy of the cell.
        s_diag2 : float
            The square of the diagonal of the cell.

        Returns
        -------
        sfx : float
            The x component of the force onto the cell from its neighbor.
        sfy : float
            The y component of the force onto the cell from its neighbor.
        sfphi : float
            The torque onto the cell from its neighbor.
        """
        # relative distance
        distance = np.sqrt(distance_sq)

        # angular relation
        dP = self.cell_phies[cell_index] - self.cell_phies[neighbor_index]
        c2dP = np.cos(dP) * np.cos(dP)

        # abbreviate some symmetric factors
        g_Lij = 1.0 / (1.0 - s_epsA2 * c2dP)
        g_Sij = (
            0.5
            * g_Lij
            * (distance_sq / s_diag2)
            * (
                1.0
                - s_epsA05
                * (
                    np.cos(2.0 * (self.cell_phies[cell_index] - alpha))
                    + np.cos(2.0 * (self.cell_phies[neighbor_index] - alpha))
                )
            )
        )
        g_Kij = g_Lij * np.exp(-self.bExp * g_Sij)

        # force onto i from j
        sfx2 = (
            g_Kij
            * distance
            * (
                np.cos(alpha)
                - s_epsA05
                * (
                    np.cos(2.0 * self.cell_phies[cell_index] - alpha)
                    + np.cos(2.0 * self.cell_phies[neighbor_index] - alpha)
                )
            )
        )
        sfy2 = (
            g_Kij
            * distance
            * (
                np.sin(alpha)
                - s_epsA05
                * (
                    np.sin(2.0 * self.cell_phies[cell_index] - alpha)
                    + np.sin(2.0 * self.cell_phies[neighbor_index] - alpha)
                )
            )
        )

        # torque onto i from j
        sfphi2 = (
            g_Kij
            * s_epsA05
            * distance_sq
            * np.sin(2.0 * (self.cell_phies[cell_index] - alpha))
        )
        sfphi2 = sfphi2 + g_Kij * s_epsA2 * s_diag2 * g_Sij * np.sin(
            2.0
            * (self.cell_phies[neighbor_index] - self.cell_phies[cell_index])
        )

        return sfx2, sfy2, sfphi2

    # ---------------------------------------------------------
    def interaction(self, cell_index: int, delta_t: float):
        """The given cell interacts with others if they are close enough.

        It describes the interaction of the cells given a force. It changes the position
        of the cell (because of a force and the intrinsic velocity) and it's angle in the
        x-y plane, phi (becuase of a torque).

        Parameters
        ----------
        cell_index : int
            The index of the cell.
        delta_t : float
            The time step.

        Returns
        -------
        dif_position : np.ndarray
            The change in the position os the cell.
        dphi : np.ndarray
            The change in the angle phi of the cell.
        -----
        """
        cell = self.cells[cell_index]

        neighbors = set(self.active_cell_indexes)
        neighbors.discard(cell_index)

        # initialization of the forces of interaction
        sfx = 0
        sfy = 0
        sfphi = 0
        # calculation of some parameters of the cell that are necessary in the interaction
        s_Rot, s_Rep, s_v0, s_DmPmS, s_SmPmS, s_epsA05, s_epsA2, s_diag2 = (
            cell.derived_parameters(self.kProp, self.kRep, self.bExp)
        )

        for neighbor_index in neighbors:
            relative_pos_x, relative_pos_y = self.relative_pos(
                cell_index, neighbor_index
            )
            # distance relative to the square and angle necessary for the interaction
            distance_sq = relative_pos_x**2 + relative_pos_y**2
            alpha = np.arctan2(relative_pos_y, relative_pos_x)

            if distance_sq < (2 * cell.major_axis) ** 2:
                sfx2, sfy2, sfphi2 = self.interaction_between_2_cells(
                    cell_index,
                    neighbor_index,
                    alpha,
                    distance_sq,
                    s_epsA05,
                    s_epsA2,
                    s_diag2,
                )
                sfx = sfx + sfx2
                sfy = sfy + sfy2
                sfphi = sfphi + sfphi2

        # forces excerted to the cell given by Grossman
        fx = s_Rep * (
            (s_SmPmS + s_DmPmS * np.cos(2 * self.cell_phies[cell_index])) * sfx
            + s_DmPmS * np.sin(2 * self.cell_phies[cell_index]) * sfy
        )
        fy = s_Rep * (
            (s_SmPmS - s_DmPmS * np.cos(2 * self.cell_phies[cell_index])) * sfy
            + s_DmPmS * np.sin(2 * self.cell_phies[cell_index]) * sfx
        )
        force = np.array([fx, fy, 0])

        # we calculate the change in the angle of orientation of the cell
        dphi = delta_t * s_Rot * sfphi

        # and in its position
        dif_position = (s_v0 * cell.direction() + force) * delta_t

        # we return the change in the position and in the phi angle of the cell
        return dif_position, dphi

    # ---------------------------------------------------------
    def move(
        self,
        dif_positions: np.ndarray,
        dphies: np.ndarray,
    ) -> None:
        """The given cell moves with a given velocity and changes its orientation.

        Attempts to move one step with a particular velocity and changes its orientation.
        If the cell arrives to a border of the culture's square, it appears on the other
        side.

        Parameters
        ----------
        dif_positions : np.ndarray
            Matrix that contains the changes in position of all the cells.
        dphies : np.ndarray
            Matrix that contains the changes in orientation of all the cells.
        -----
        """
        # Updating the cell's position
        self.cell_positions = self.cell_positions + dif_positions

        # and the velocity
        self.cell_phies = self.cell_phies + dphies

        # Enforcing boundary condition
        self.cell_positions = np.mod(self.cell_positions, self.side)

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
                    l = self.side
                    # choose a random position and angle in the xy plane (phi)
                    Cell(
                        position=np.array(
                            [
                                self.rng.uniform(low=0, high=l),
                                self.rng.uniform(low=0, high=l),
                                0,
                            ]
                        ),
                        culture=self,
                        is_stem=self.first_cell_is_stem,
                        phi=self.rng.uniform(low=0, high=2 * np.pi),
                        parent_index=0,
                        available_space=True,
                    )
        # Save the data (for dat, ovito, and/or SQLite)
        self.output.record_culture_state(
            tic=0,
            cells=self.cells,
            cell_positions=self.cell_positions,
            active_cell_indexes=self.active_cell_indexes,
            side=self.side,
        )
        # we simulate for num_times time steps
        reproduction = self.reproduction
        movement = self.movement
        # time parameters for movement and saving
        t = 0 
        delta_t = 0.1
        save_t = 0.5
        tolerance_t = 1e-2

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
                dif_positions = np.empty((0, 3), float)
                dphies = np.array([])
                for index in self.active_cell_indexes:
                    dif_position, dphi = self.interaction(
                        cell_index=index, delta_t=delta_t
                    )
                    dif_positions = np.append(
                        dif_positions, [dif_position], axis=0
                    )
                    dphies = np.append(dphies, dphi)

                self.move(dif_positions=dif_positions, dphies=dphies)
                t = t + delta_t
            
            # Save the data (for dat, ovito, and/or SQLite)
            # we save it when the time step is a multiple of save_t using the tolerance
            if np.mod(t, save_t)<tolerance_t or np.mod(t, save_t)>save_t-tolerance_t:
                self.output.record_culture_state(
                    tic=i,
                    cells=self.cells,
                    cell_positions=self.cell_positions,
                    active_cell_indexes=self.active_cell_indexes,
                    side=self.side,
                )
