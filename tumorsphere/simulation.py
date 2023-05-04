import matplotlib.pyplot as plt

from tumorsphere.cells import *
from tumorsphere.culture import *


class Simulation:
    def __init__(
        self,
        first_cell_is_stem=True,
        prob_stem=[0.36],  # Wang HARD substrate value
        prob_diff=[0],  # p_d; probability that a CSC gives a DCC and then
        # loses stemness (i.e. prob. that a CSC gives two DCCs)
        num_of_realizations=10,
        num_of_steps_per_realization=10,
        rng_seed=0x87351080E25CB0FAD77A44A3BE03B491,
        cell_radius=1,
        adjacency_threshold=4,  # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        cell_max_repro_attempts=1000,
        continuous_graph_generation=False,
        # THE USER MUST PROVIDE A HIGH QUALITY SEED
        # in spite of the fact that I set a default
        # (so the code doesn't break e.g. when testing)
    ):
        # main simulation attributes
        self.first_cell_is_stem = first_cell_is_stem
        self.prob_stem = prob_stem
        self.prob_diff = prob_diff
        self.num_of_realizations = num_of_realizations
        self.num_of_steps_per_realization = num_of_steps_per_realization
        self._rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

        # dictionary storing the culture objects
        self.cultures = {}

        # dictionary storing simulation data (same keys as the previous one)
        self.data = {}

        # array with the average evolution over realizations, per p_s, per p_d
        self.average_data = {}
        for pd in self.prob_diff:
            for ps in self.prob_stem:
                self.average_data[f"average_pd={pd}_ps={ps}"] = {
                    "total": np.zeros(self.num_of_steps_per_realization),
                    "active": np.zeros(self.num_of_steps_per_realization),
                    "total_stem": np.zeros(self.num_of_steps_per_realization),
                    "active_stem": np.zeros(self.num_of_steps_per_realization),
                }

        # attributes to pass to the culture (and cells)
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.continuous_graph_generation = continuous_graph_generation

    def simulate(self):
        for k in range(len(self.prob_diff)):
            for i in range(len(self.prob_stem)):
                for j in range(self.num_of_realizations):
                    # we compute a string with the ps and number of this realization
                    current_realization_name = f"culture_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}_realization_{j}"
                    # we instantiate the culture of this realization as an item of
                    # the self.cultures dictionary, with the string as key
                    self.cultures[current_realization_name] = Culture(
                        adjacency_threshold=self.adjacency_threshold,
                        cell_radius=self.cell_radius,
                        cell_max_repro_attempts=self.cell_max_repro_attempts,
                        first_cell_is_stem=self.first_cell_is_stem,
                        prob_stem=self.prob_stem[i],
                        prob_diff=self.prob_diff[k],  # implementar en culture
                        continuous_graph_generation=self.continuous_graph_generation,
                        rng_seed=self.rng.integers(low=2**20, high=2**50),
                    )
                    # we simulate the culture's growth and retrive data in the
                    # self.data dictionary, with the same string as key
                    self.data[current_realization_name] = self.cultures[
                        current_realization_name
                    ].simulate_with_data(self.num_of_steps_per_realization)
                # now we compute the averages
                self.average_data[
                    f"average_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}"
                ] = self._average_of_data_ps_i_and_pd_k(i, k)

        # picklear los objetos tiene que ser responsabilidad del método
        # simulate de culture, ya que es algo que se hace en medio de la
        # evolución, pero va a necesitar que le pase el current_realization_name
        # para usarlo como nombre del archivo

    def _average_of_data_ps_i_and_pd_k(self, i, k):
        # For prob_stem[i] and prob_diff[k], we average
        # over the j realizations (m is a string)
        data_of_ps_i_and_pd_k_realizations = {}
        for j in range(self.num_of_realizations):
            data_of_ps_i_and_pd_k_realizations[j] = self.data[
                f"culture_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}_realization_{j}"
            ]

        # we stack the data for all variables and average it:
        # total
        vstacked_total = data_of_ps_i_and_pd_k_realizations[0]["total"]
        for j in range(1, self.num_of_realizations):
            vstacked_total = np.vstack(
                (
                    vstacked_total,
                    data_of_ps_i_and_pd_k_realizations[j]["total"],
                )
            )
        average_total = np.mean(
            vstacked_total,
            axis=0,
        )

        # active
        vstacked_active = data_of_ps_i_and_pd_k_realizations[0]["active"]
        for j in range(1, self.num_of_realizations):
            vstacked_active = np.vstack(
                (
                    vstacked_active,
                    data_of_ps_i_and_pd_k_realizations[j]["active"],
                )
            )
        average_active = np.mean(
            vstacked_active,
            axis=0,
        )

        # total_stem
        vstacked_total_stem = data_of_ps_i_and_pd_k_realizations[0][
            "total_stem"
        ]
        for j in range(1, self.num_of_realizations):
            vstacked_total_stem = np.vstack(
                (
                    vstacked_total_stem,
                    data_of_ps_i_and_pd_k_realizations[j]["total_stem"],
                )
            )
        average_total_stem = np.mean(
            vstacked_total_stem,
            axis=0,
        )

        # active_stem
        vstacked_active_stem = data_of_ps_i_and_pd_k_realizations[0][
            "active_stem"
        ]
        for j in range(1, self.num_of_realizations):
            vstacked_active_stem = np.vstack(
                (
                    vstacked_active_stem,
                    data_of_ps_i_and_pd_k_realizations[j]["active_stem"],
                )
            )
        average_active_stem = np.mean(
            vstacked_active_stem,
            axis=0,
        )

        # we organaize data in the appropriate format for storing in
        # self.average_data[f"average_pd={self.prob_diff[k]}_ps={self.prob_stem[i]}"]
        average = {
            "total": average_total,
            "active": average_active,
            "total_stem": average_total_stem,
            "active_stem": average_active_stem,
        }
        return average

    def plot_average_data(self, ps_index, pd_index):
        # create a figure and axis objects
        fig, ax = plt.subplots()

        # plot each row of the array with custom labels and colors
        data = self.average_data[
            f"average_pd={self.prob_diff[pd_index]}_ps={self.prob_stem[ps_index]}"
        ]

        ax.plot(data["total"], label="Total", color="blue")
        ax.plot(
            data["active"],
            label="Total active",
            color="green",
        )
        ax.plot(data["total_stem"], label="Stem", color="orange")
        ax.plot(data["active_stem"], label="Active stem", color="red")

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
