from tumorsphere.cells import *


class Culture:
    def __init__(
        self,
        adjacency_threshold=np.sqrt(2) / 2,
        cell_radius=1,
        cell_max_repro_attempts=10000,
        first_cell_type="cell",
        prob_stem=0.36,  # Wang HARD substrate value
    ):
        self.cell_max_repro_attempts = cell_max_repro_attempts
        self.adjacency_threshold = adjacency_threshold
        self.cell_radius = cell_radius
        self.prob_stem = prob_stem
        if first_cell_type == "cell":
            first_cell_object = Cell(
                position=np.array([0, 0, 0]),
                culture=self,
                adjacency_threshold=self.adjacency_threshold,
                radius=self.cell_radius,
                max_repro_attempts=cell_max_repro_attempts,
            )
        elif first_cell_type == "dcc":
            first_cell_object = Dcc(
                position=np.array([0, 0, 0]),
                culture=self,
                adjacency_threshold=self.adjacency_threshold,
                radius=self.cell_radius,
                max_repro_attempts=cell_max_repro_attempts,
            )
        elif first_cell_type == "csc":
            first_cell_object = Csc(
                position=np.array([0, 0, 0]),
                culture=self,
                adjacency_threshold=self.adjacency_threshold,
                radius=self.cell_radius,
                max_repro_attempts=cell_max_repro_attempts,
                prob_stem=self.prob_stem,
            )
        else:
            raise ValueError(
                "Cell types accepted are 'cell', 'dcc' and 'csc'."
            )

        self.cells = [first_cell_object]
        self.graph = nx.Graph()
        self.graph.add_node(first_cell_object)

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
            ax.scatter(x, y, z, c=cell.color, marker="o")

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x, sphere_y, sphere_z, color=cell.color, alpha=0.2
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
            ax.scatter(x, y, z, c=cell.color, marker="o")

            # plot a sphere at the position of the cell
            u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
            sphere_x = cell.position[0] + np.cos(u) * np.sin(v) * cell.radius
            sphere_y = cell.position[1] + np.sin(u) * np.sin(v) * cell.radius
            sphere_z = cell.position[2] + np.cos(v) * cell.radius
            ax.plot_surface(
                sphere_x, sphere_y, sphere_z, color=cell.color, alpha=0.2
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
            cells = random.sample(self.cells, k=len(self.cells))
            # I had to point to the cells in a copied list,
            # if not, strange things happened
            for cell in cells:
                cell.reproduce()
