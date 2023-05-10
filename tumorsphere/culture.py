"""
Module containing the Culture class.

Classes:
    - Culture: Class that represents a culture of cells. Usually dependent
    on the Simulation class.
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time

from tumorsphere.cells import Cell


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
