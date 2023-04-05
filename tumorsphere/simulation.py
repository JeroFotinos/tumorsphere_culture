from tumorsphere.cells import *
from tumorsphere.culture import *

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
