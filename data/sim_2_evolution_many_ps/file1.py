# ========================================================================
# CELLS.PY
# ========================================================================

import copy
import random
import time

import matplotlib.pyplot as plt

# import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

colors = {True: "red", False: "blue"}

# probabilities = {'ps' : 0.36, 'pd' : 0.16}
# prob_stem = 0.36


class Cell:
    def __init__(
        self,
        position,
        culture,
        adjacency_threshold=4,  # upper bound to third neighbor distance in HCP
        radius=1,
        is_stem=False,
        max_repro_attempts=10000,
        prob_stem=0.36,  # Wang HARD substrate value
        continuous_graph_generation=False,
        rng_seed=23978461273864,
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
        self._swap_probability = 0.5

        # We instantiate the cell's RNG with the entropy provided
        self.rng = np.random.default_rng(rng_seed)

        # Plotting and graph related attributes
        self._continuous_graph_generation = continuous_graph_generation
        self._colors = {True: "red", False: "blue"}

        # Attributes that evolve with the simulation
        self.neighbors = []
        self.available_space = True
        self.is_stem = is_stem

    def find_neighbors_from_entire_culture_from_scratch(self):
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
        # self.neighbors = []
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
        # if the cell is a newborn, it will only have its parent as neighbor,
        # so neighbors of its neighbors are just the neighbors of its parent.
        # The first time we have to go a level deeper.
        if len(self.neighbors) < 20:
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
        # if the cell is a newborn, it will only have its parent as neighbor,
        # so neighbors of its neighbors are just the neighbors of its parent.
        # The first time we have to go a level deeper.
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
        theta = np.random.uniform(low=0, high=2 * np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        x = 2 * self.radius * np.sin(phi) * np.cos(theta)
        y = 2 * self.radius * np.sin(phi) * np.sin(theta)
        z = 2 * self.radius * np.cos(phi)
        new_position = self.position + np.array([x, y, z])
        return new_position

    def reproduce(self):
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
                    if self.rng.random() <= self.prob_stem:
                        child_cell = Cell(
                            position=child_position,
                            culture=self.culture,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=True,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
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
                            continuous_graph_generation=self._continuous_graph_generation,
                            rng_seed=self.rng.integers(
                                low=2**20, high=2**50
                            ),
                        )
                        if self.rng.random() <= self._swap_probability:
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
    def __init__(
        self,
        adjacency_threshold=4,
        cell_radius=1,
        cell_max_repro_attempts=10000,
        first_cell_is_stem=False,
        prob_stem=0.36,  # Wang HARD substrate value
        continuous_graph_generation=False,
        rng_seed=110293658491283598,
        # THE SIMULATION MUST PROVIDE A SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # attributes to inherit to the cells
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem

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
            continuous_graph_generation=continuous_graph_generation,
            rng_seed=self.rng.integers(low=2**20, high=2**50),
        )

        # we initialize the lists and graphs with the first cell
        self.cells = [first_cell_object]
        self.active_cells = [first_cell_object]
        # self.graph = nx.Graph()
        # self.graph.add_node(first_cell_object)

    def plot_culture_dots(self):
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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for cell in self.cells:
            x, y, z = cell.position
            ax.scatter(x, y, z, c=cell._colors[cell.is_stem], marker="o")

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=cell._colors[cell.is_stem],
                alpha=0.2,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.mouse_init()  # initialize mouse rotation

        plt.show()

    def plot_culture_fig(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for cell in self.cells:
            x, y, z = cell.position
            ax.scatter(x, y, z, c=cell._colors[cell.is_stem], marker="o")

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x,
                sphere_y,
                sphere_z,
                color=cell._colors[cell.is_stem],
                alpha=0.2,
            )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # ax.mouse_init()  # initialize mouse rotation

        # plt.show()
        return fig

    def plot_graph(self):
        nx.draw(self.graph)

    # to be implemented

    def generate_adjacency_graph_from_scratch(self):
        self.graph = nx.Graph()
        for cell in self.cells:
            self.graph.add_node(cell)
        for i, cell1 in enumerate(self.cells):
            for cell2 in self.cells[i + 1 :]:
                if cell2 in cell1.neighbors:
                    self.graph.add_edge(cell1, cell2)

    def simulate(self, num_times):
        for i in range(num_times):
            cells = self.rng.permutation(self.active_cells)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for cell in cells:
                cell.reproduce()

    def any_csc_in_culture_boundary(self):
        in_boundary = [(cell.available_space) for cell in self.active_cells]
        any_csc_in_boundary = np.any(in_boundary)
        return any_csc_in_boundary

    def simulate_with_data(self, num_times):
        # we initialize the arrays that will make up the data
        total = np.zeros(num_times)
        active = np.zeros(num_times)
        total_stem = np.zeros(num_times)
        active_stem = np.zeros(num_times)

        # we count the initial amount of CSCs
        if self.first_cell_is_stem:
            initial_amount_of_csc = 1
        else:
            initial_amount_of_csc = 0

        # we asign the initial values for the data
        total[0] = 1
        active[0] = 1
        total_stem[0] = initial_amount_of_csc
        active_stem[0] = initial_amount_of_csc

        # print data
        with open("data.txt", "a") as file:
            file.write(
                f"{total[0]}, {active[0]}, {total_stem[0]}, {active_stem[0]} \n"
            )

        # we simulate for num_times time steps
        for i in range(num_times):
            #### MEASURING TIME

            # # start the timer
            # start_time = time.perf_counter()

            #### CODE

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
            total[i] = len(self.cells)
            active[i] = len(self.active_cells)
            total_stem[i] = total_stem_counter
            active_stem[i] = active_stem_counter

            # we stack the arrays that make up the data into a single array to be returned
            data = np.vstack((total, active, total_stem, active_stem))

            #### MEASURING TIME

            # # stop the timer
            # end_time = time.perf_counter()

            # # calculate the elapsed time
            # elapsed_time = end_time - start_time

            # # print the elapsed time
            # with open("filename.txt", "a") as file:
            #     file.write(f"Time taken: {elapsed_time:.6f} seconds" + "\n")

            # print data
            with open("data/val_ps-val_realiz.dat", "a") as file:
                file.write(
                    f"{total[i]}, {active[i]}, {total_stem[i]}, {active_stem[i]} \n"
                )

        return data


# ========================================================================
# SIMULATION.PY
# ========================================================================

import matplotlib.pyplot as plt


class Simulation:
    def __init__(
        self,
        first_cell_is_stem=True,
        prob_stem=[0.36],  # Wang HARD substrate value
        num_of_realizations=10,
        num_of_steps_per_realization=10,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius=1,
        adjacency_threshold=4,
        cell_max_repro_attempts=5000,
        continuous_graph_generation=False,
        # THE USER MUST PROVIDE A HIGH QUALITY SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # main simulation attributes
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self.rng = np.random.default_rng(rng_seed)

        # dictionary storing the culture objects
        self.cultures = {}

        # dictionary storing simulation data (same keys as the previous one)
        self.data = {}

        # array with the average evolution over realizations, per p_s
        self.average_data = np.zeros(
            shape=(4, num_of_steps_per_realization, len(prob_stem))
        )
        # está medio nasty así (me gustaría no ser tan específico, ni llenar de
        # ceros al pedo), corregir en algún momento

        # attributes to pass to the culture (and cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.continuous_graph_generation = continuous_graph_generation

    def simulate(self):
        for i in range(len(self.prob_stem)):
            for j in range(self.num_of_realizations):
                # we compute a string with the ps and number of this realization
                current_realization_name = (
                    f"culture_ps={self.prob_stem[i]}_realization_{j}"
                )
                # we instantiate the culture of this realization as an item of
                # the self.cultures dictionary, with the string as key
                self.cultures[current_realization_name] = Culture(
                    adjacency_threshold=self.adjacency_threshold,
                    cell_radius=self.cell_radius,
                    cell_max_repro_attempts=self.cell_max_repro_attempts,
                    first_cell_is_stem=self.first_cell_is_stem,
                    prob_stem=self.prob_stem[i],
                    continuous_graph_generation=self.continuous_graph_generation,
                    rng_seed=self.rng.integers(low=2**20, high=2**50),
                )
                # we simulate the culture's growth and retrive data in the
                # self.data dictionary, with the same string as key
                self.data[current_realization_name] = self.cultures[
                    current_realization_name
                ].simulate_with_data(self.num_of_steps_per_realization)
            # now we compute the averages
            # self.average_data[:, :, i] = np.vstack((self.average_of_data_row_k_and_ps_i(k, i) for k in range(4)))
            self.average_data[:, :, i] = np.vstack(
                (
                    self.average_of_data_row_k_and_ps_i(0, i),
                    self.average_of_data_row_k_and_ps_i(1, i),
                    self.average_of_data_row_k_and_ps_i(2, i),
                    self.average_of_data_row_k_and_ps_i(3, i),
                )
            )

        # picklear los objetos tiene que ser responsabilidad del método
        # simulate de culture, ya que es algo que se hace en medio de la
        # evolución, pero va a necesitar que le pase el current_realization_name
        # para usarlo como nombre del archivo

    def average_of_data_row_k_and_ps_i(self, k, i):
        # For prob_stem[i]
        # k=0 --> average of total
        # k=1 --> average of active
        # k=2 --> average of total stem
        # k=3 --> average of active stem
        vstacked_data = self.data[
            f"culture_ps={self.prob_stem[i]}_realization_{0}"
        ][k]
        for j in range(1, self.num_of_realizations):
            vstacked_data = np.vstack(
                (
                    vstacked_data,
                    self.data[
                        f"culture_ps={self.prob_stem[i]}_realization_{j}"
                    ][k],
                )
            )

        average_row_k = np.mean(
            vstacked_data,
            axis=0,
        )
        return average_row_k

    def plot_average_data(self, ps_index):
        # create a figure and axis objects
        fig, ax = plt.subplots()

        # plot each row of the array with custom labels and colors
        ax.plot(self.average_data[0, :, ps_index], label="Total", color="blue")
        ax.plot(
            self.average_data[1, :, ps_index],
            label="Total active",
            color="green",
        )
        ax.plot(
            self.average_data[2, :, ps_index], label="Stem", color="orange"
        )
        ax.plot(
            self.average_data[3, :, ps_index], label="Active stem", color="red"
        )

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
# PROGRAMA.PY
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

import pickle

# open a file to save the pickled dictionary
with open("sim_average_data.pkl", "wb") as f:
    pickle.dump(sim.average_data, f)

# open a file to save the pickled dictionary
with open("sim_data.pkl", "wb") as f:
    pickle.dump(sim.data, f)
