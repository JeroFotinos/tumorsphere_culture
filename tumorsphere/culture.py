"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""
import sqlite3
from datetime import datetime
from typing import Set

import numpy as np

from tumorsphere.cells import Cell


class Culture:
    def __init__(
        self,
        adjacency_threshold: float = 4,
        cell_radius: float = 1,
        cell_max_repro_attempts: int = 1000,
        first_cell_is_stem: bool = True,
        prob_stem: float = 0,
        prob_diff: float = 0,
        rng_seed: int = 110293658491283598,
        swap_probability: float = 0.5,
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
        rng : numpy.random.Generator
            Random number generator.
        first_cell_is_stem : bool
            Whether the first cell is a stem cell or not.
        cell_positions : numpy.ndarray
            Matrix to store the positions of all cells in the culture.
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

        # we instantiate the culture's RNG with the entropy provided
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # state whether this is a csc-seeded culture
        self.first_cell_is_stem = first_cell_is_stem

        # initialize the positions matrix
        self.cell_positions = np.empty((0, 3), float)

        # we initialize the lists of cells
        self.cells = []
        self.active_cell_indexes = []

        # time at wich the culture was created
        self.simulation_start = self._get_simulation_time()

        # connection to the SQLite database (defined while simulating)
        self.conn = None 


    # ----------------database related behavior----------------

    def _get_simulation_time(self):
        # we get the current date and time
        current_time = datetime.now()
        # we format the string
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return time_string

    def _create_tables(self):
        with self.conn:
            cursor = self.conn.cursor()

            # Enable foreign key constraints for this connection
            cursor.execute("PRAGMA foreign_keys = ON;")

            # Creating the Culture table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Cultures (
                    culture_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prob_stem REAL NOT NULL,
                    prob_diff REAL NOT NULL,
                    culture_seed INTEGER NOT NULL,
                    simulation_start TIMESTAMP NOT NULL,
                    adjacency_threshold REAL NOT NULL,
                    swap_probability REAL NOT NULL
                );
            """
            )

            # Creating the Cells table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS Cells (
                _index INTEGER PRIMARY KEY,
                parent_index INTEGER,
                position_x REAL NOT NULL,
                position_y REAL NOT NULL,
                position_z REAL NOT NULL,
                t_creation INTEGER NOT NULL,
                t_deactivation INTEGER,
                culture_id INTEGER,
                FOREIGN KEY(culture_id) REFERENCES Cultures(culture_id)
            );
            """
            )

            # Creating the StemChange table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS StemChanges (
                change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                cell_id INTEGER NOT NULL,
                t_change INTEGER NOT NULL,
                is_stem BOOLEAN NOT NULL,
                FOREIGN KEY(cell_id) REFERENCES Cells(_index)
            );
            """
            )

    def _insert_culture_record_and_get_culture_id(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO Cultures (prob_stem, prob_diff, culture_seed, simulation_start, adjacency_threshold, swap_probability)
                VALUES (?, ?, ?, ?, ?, ?);
            """,
                (
                    self.prob_stem,
                    self.prob_diff,
                    int(self.rng_seed),
                    self.simulation_start,
                    self.adjacency_threshold,
                    self.swap_probability,
                ),
            )
            culture_id = cursor.lastrowid
            return culture_id
    
    def record_stemness(self, cell_index, tic):
        with self.conn:
            cursor = self.conn.cursor()
            stemness = self.cells[cell_index].is_stem
            cursor.execute(
                """
                INSERT INTO StemChanges (cell_id, t_change, is_stem)
                VALUES (?, ?, ?);
            """,
            (
                int(cell_index),
                tic,
                stemness,
            ),
            )

    def record_deactivation(self, cell_index, tic):
        with self.conn:
            cursor = self.conn.cursor()

            # Recording (updating) the t_deactivation value for the specified cell
            cursor.execute(
                """
                UPDATE Cells
                SET t_deactivation = ?
                WHERE _index = ?;
                """,
                (tic, int(cell_index))
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
                    - self.cell_positions[a_cell._index]
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
        # theta = np.random.uniform(low=0, high=2 * np.pi)
        # phi = np.random.uniform(low=0, high=np.pi)
        # theta = self.rng.uniform(low=0, high=2 * np.pi)
        # , size=number_of_points
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
        find a position that doesn't overlap with existing cells, for the
        estabished maximum number of attempts, no new cell is created.

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
                            self.record_stemness(cell_index, tic)
                        elif (
                            self.rng.random() <= self.swap_probability
                        ):  # pa = 1-ps-pd
                            cell.is_stem = False
                            self.record_stemness(cell_index, tic)
                            child_cell.is_stem = True
                            self.record_stemness(child_cell._index, tic)
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
                cell.available_space = False
                self.active_cell_indexes.remove(cell_index)
                self.record_deactivation(cell_index, tic)
                # if there was no available space, we turn off reproduction
                # and record the change in the Cells table of the DataBase
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing
        # (reproduction is turned off)

    # ---------------------------------------------------------

    def simulate(self, num_times: int, culture_name: str) -> None:
        """Docstring."""

        # connection to the SQLite database
        self.conn = sqlite3.connect(f"data/{culture_name}.db")

        # we create the tables of the DB
        self._create_tables()

        # we insert the register corresponding to this culture
        self.culture_id = self._insert_culture_record_and_get_culture_id()

        # we instantiate the first cell
        first_cell_object = Cell(
            position=np.array([0, 0, 0]),
            culture=self,
            is_stem=self.first_cell_is_stem,
            parent_index=0,
            available_space=True,
        )

        # we simulate for num_times time steps
        for i in range(1, num_times):
            # we get a permuted copy of the cells list
            active_cell_indexes = self.rng.permutation(self.active_cell_indexes)
            # and reproduce the cells in this random order
            for index in active_cell_indexes:
                self.reproduce(cell_index=index, tic=i)

    def simulate_with_dat_files(
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
            culture_pd={sim.prob_diff[k]}_ps={sim.prob_stem[i]}_rng_seed={seed}.dat
        """

        # we instantiate the first cell
        first_cell_object = Cell(
            position=np.array([0, 0, 0]),
            culture=self,
            is_stem=self.first_cell_is_stem,
            parent_index=0,
            available_space=True,
        )

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
            active_cell_indexes = self.rng.permutation(self.active_cell_indexes)
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
            for index in self.active_cell_indexes:
                if self.cells[index].is_stem:
                    active_stem_counter = active_stem_counter + 1

            # we save the data to a file
            with open(f"data/{culture_name}.dat", "a") as file:
                file.write(
                    f"{len(self.cells)}, {len(self.active_cell_indexes)}, {total_stem_counter}, {active_stem_counter} \n"
                )

    def simulate_with_ovito_data(
        self, num_times: int, culture_name: str
    ) -> None:
        """Idem to simulate_with_persistent_data, but saving data for plotting
        with ovito.

        Parameters
        ----------
        num_times : int
            The number of time steps to simulate the cellular automaton.
        culture_name : str
            The name of the culture in the simulation, in the format
            culture_pd={sim.prob_diff[k]}_ps={sim.prob_stem[i]}_rng_seed={seed}.dat
        """

        # we instantiate the first cell
        first_cell_object = Cell(
            position=np.array([0, 0, 0]),
            culture=self,
            is_stem=self.first_cell_is_stem,
            parent_index=0,
            available_space=True,
        )

        # we simulate for num_times time steps
        for i in range(1, num_times):
            # we get a permuted copy of the cells list
            active_cell_indexes = self.rng.permutation(self.active_cell_indexes)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for index in active_cell_indexes:
                self.reproduce(index)

            # we save the data for ovito
            self.make_ovito_data_file(t=i, culture_name=culture_name)

    def make_ovito_data_file(self, t, culture_name):
        """Writes the data file in path for ovito, for time step t of self.
        Auxiliar function for simulate_with_ovito_data.
        """
        path_to_write = f"ovito_data_{culture_name}.{t:03}"
        with open(path_to_write, "w") as file_to_write:
            file_to_write.write(str(len(self.cells)) + "\n")
            file_to_write.write(
                ' Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"Properties=species:S:1:pos:R:3:Color:r:1'
                + "\n"
            )

            for cell in self.cells:  # csc activas
                if cell.is_stem and cell.available_space:
                    line = (
                        "active_stem "
                        + str(self.cell_positions[cell._index][0])
                        + " "
                        + str(self.cell_positions[cell._index][1])
                        + " "
                        + str(self.cell_positions[cell._index][2])
                        + " "
                        + "1"
                        + "\n"
                    )
                    file_to_write.write(line)

            for cell in self.cells:  # csc quiesc
                if cell.is_stem and (not cell.available_space):
                    line = (
                        "quiesc_stem "
                        + str(self.cell_positions[cell._index][0])
                        + " "
                        + str(self.cell_positions[cell._index][1])
                        + " "
                        + str(self.cell_positions[cell._index][2])
                        + " "
                        + "2"
                        + "\n"
                    )
                    file_to_write.write(line)

            for cell in self.cells:  # dcc activas
                if (not cell.is_stem) and cell.available_space:
                    line = (
                        "active_diff "
                        + str(self.cell_positions[cell._index][0])
                        + " "
                        + str(self.cell_positions[cell._index][1])
                        + " "
                        + str(self.cell_positions[cell._index][2])
                        + " "
                        + "3"
                        + "\n"
                    )
                    file_to_write.write(line)

            for cell in self.cells:  # dcc quiesc
                if not (cell.is_stem or cell.available_space):
                    line = (
                        "quiesc_diff "
                        + str(self.cell_positions[cell._index][0])
                        + " "
                        + str(self.cell_positions[cell._index][1])
                        + " "
                        + str(self.cell_positions[cell._index][2])
                        + " "
                        + "4"
                        + "\n"
                    )
                    file_to_write.write(line)
