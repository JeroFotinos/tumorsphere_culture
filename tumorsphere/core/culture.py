"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""

# import os
from datetime import datetime

import pandas as pd
import numpy as np

from tumorsphere.core.cells import Cell
from tumorsphere.core.output import TumorsphereOutput
from tumorsphere.core.spatial_hash_grid import SpatialHashGrid


class Culture:
    """
    Class that represents a culture of cells.

    This class handles the simulation, as well as some behavior of the cells,
    such as reproduction.
    """

    def __init__(
        self,
        output: TumorsphereOutput,
        grid: SpatialHashGrid,
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
        output : TumorsphereOutput
            The output object to record the simulation data.
        grid : SpatialHashGrid
            The spatial hash grid to be used in the simulation.
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

        # we instantiate the culture's RNG with the provided entropy
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

        # Additional objects
        self.output = output
        self.grid = grid

        # we set the grid's culture to this one
        self.grid.culture = self

    # ----------------database related behavior----------------

    def _get_simulation_time(self):
        # we get the current date and time
        current_time = datetime.now()
        # we format the string
        time_string = current_time.strftime("%Y-%m-%d %H:%M:%S")
        return time_string

    # ------------------cell related behavior------------------

    def generate_new_position(self, cell_index: int) -> np.ndarray:
        """
        Generate a proposed position for the child, adjacent to the given one.

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

                # if the position is not within the bounds of the simulation
                # we get the corresponding position
                if not self.grid.is_position_in_bounds(child_position):
                    child_position = self.grid.get_in_bounds_position(
                        child_position
                    )

                # set of all existing cell indexes that would neighbor the new
                # cell
                neighbor_indices = list(
                    self.grid.find_neighbors(
                        position=child_position,
                    )
                )
                # modifies the set in-place to remove the parent cell index
                neighbor_indices.remove(cell_index)

                # array with the distances from the proposed child position to
                # the other cells
                if len(neighbor_indices) > 0:
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

    # --------------------------- Radiotherapy things ------------------------

    def realization_name(self) -> str:
        """Return the name of the realization."""
        name = (
            f"culture_pd={self.prob_diff}"
            f"_ps={self.prob_stem}"
            f"_rng_seed={self.rng_seed}"
        )
        return name

    def radiotherapy_w_susceptibility(self) -> None:
        """Simulate a radiotherapy session by assigning susceptibilities.

        This function simulates a radiotherapy session where, due to increased
        O2 consumption, the active cells are more sensitive to radiation than
        quiescent cells. The probability of survival is different for active
        and quiescent cells, by a factor beta. However, all of this is left
        for postprocessing, so data can be used both for the described
        situation, or for another one where the cells are killed with a
        probability that varies with their position.

        A pandas.DataFrame is generated and saved with the following columns:
        - the norm of the position of the cell
        - the cell's stemness
        - whether the cell is active
        - a “suceptibility” that will indicate whether the cell was killed
          given the survival ratio (in postprocessing).
        """
        # we make the dictionary for the dataframe that will store the data
        susceptibility = self.rng.random(size=len(self.cells))
        norms = np.linalg.norm(self.cell_positions, axis=1)
        data = {
            "position_norm": norms,
            "stemness": [],
            "active": [],
            "susceptibility": susceptibility,
        }

        # we get the stemness, activity, and killing status of the cells
        for cell in self.cells:
            data["stemness"].append(cell.is_stem)
            data["active"].append(cell._index in self.active_cell_indexes)
            assert (
                cell._index in self.active_cell_indexes
            ) == cell.available_space

        # we make the dataframe
        df = pd.DataFrame(data, index=False)

        # we save the dataframe to a file
        filename = (
            f"radiotherapy_active_targeted_{self.realization_name()}.csv"
        )
        df.to_csv(filename)

    def simulate(self, num_times: int) -> None:
        """Simulate culture growth for a specified number of time steps.

        At each time step, we randomly sort the list of active cells and then
        we tell them to reproduce one by one.

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

            # we instantiate the first cell
            Cell(
                position=np.array([0, 0, 0]),
                culture=self,
                is_stem=self.first_cell_is_stem,
                parent_index=0,
                available_space=True,
            )

        # Save the data (for dat, ovito, and/or SQLite)
        self.output.record_culture_state(
            tic=0,
            cells=self.cells,
            cell_positions=self.cell_positions,
            active_cell_indexes=self.active_cell_indexes,
        )

        # we simulate for num_times time steps
        for i in range(1, num_times + 1):
            # we get a permuted copy of the cells list
            active_cell_indexes = self.rng.permutation(
                self.active_cell_indexes
            )
            # and reproduce the cells in this random order
            for index in active_cell_indexes:
                self.reproduce(cell_index=index, tic=i)

            # Save the data (for dat, ovito, and/or SQLite)
            self.output.record_culture_state(
                tic=i,
                cells=self.cells,
                cell_positions=self.cell_positions,
                active_cell_indexes=self.active_cell_indexes,
            )

        self.output.record_final_state(
            tic=num_times,
            cells=self.cells,
            cell_positions=self.cell_positions,
            active_cell_indexes=self.active_cell_indexes,
        )
