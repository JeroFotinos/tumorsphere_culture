from tumorsphere.cells import *


class Culture:
    def __init__(
        self,
        adjacency_threshold=4, # 2.83 approx 2*np.sqrt(2), hcp second neighbor distance
        cell_radius=1,
        cell_max_repro_attempts=1000,
        first_cell_is_stem=False,
        prob_stem=0.36,  # Wang HARD substrate value
        prob_diff = 0,
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
            prob_diff = self.prob_diff,
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
            ax.scatter(x, y, z, c=cell._colors[(cell.is_stem, cell in self.active_cells)], marker="o")

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
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        for cell in self.cells:
            x, y, z = cell.position
            ax.scatter(x, y, z, c=cell._colors[(cell.is_stem, cell in self.active_cells)], marker="o")

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

    # ====================================================================

    def simulate(self, num_times):
        for i in range(num_times):
            cells = self.rng.permutation(self.active_cells)
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for cell in cells:
                cell.reproduce()

    def any_csc_in_culture_boundary(self):
        stem_in_boundary = [(cell.available_space and cell.is_stem) for cell in self.active_cells]
        any_csc_in_boundary = np.any(stem_in_boundary)
        return any_csc_in_boundary

        

    def simulate_with_data(self, num_times):
        
        # we use a dictionary to store the data arrays and initialize them
        data = {
            "total" : np.zeros(num_times),
            "active" : np.zeros(num_times),
            "total_stem" : np.zeros(num_times),
            "active_stem" : np.zeros(num_times),
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