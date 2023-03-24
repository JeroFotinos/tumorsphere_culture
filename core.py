import numpy as np
import networkx as nx


class Cell:
    def __init__(self, position, culture, adjacency_threshold=2, radius=1, max_repro_attempts=10000):
        self.position = position
        # NumPy array, vector with 3 components
        self.culture = culture
        self.adjacency_threshold = adjacency_threshold
        # distance to second neighbor in HPC
        self.radius = radius  # radius of cell
        self.max_repro_attempts = max_repro_attempts
        self.neighbors = []
        self.available_space=True #apagar

    # TO DO: a find_neighbors_from_neighbors method that
    # looks for neighbors of a cell in their neighbors and
    # their neighbor's neighbors. This will speed things up.
    # We can implement a test that checks that the result is
    # the same.

    def find_neighbors_from_scratch(self):
        self.neighbors = []
        # si las células se mueven, hay que calcular toda la lista de cero
        for cell in self.culture.cells:
            if (cell is not self and cell not in self.neighbors) and np.linalg.norm(
                self.position - cell.position
            ) < self.adjacency_threshold:
                self.neighbors.append(cell)
    
    def find_neighbors(self):
        # self.neighbors = []
        # como las células no se mueven, sólo se pueden agregar vecinos, por
        # lo que no hay necesidad de reiniciar la lista, sólo añadimos
        # los posibles nuevos vecinos
        for cell in self.culture.cells:
            if (cell is not self and cell not in self.neighbors) and np.linalg.norm(
                self.position - cell.position
            ) < self.adjacency_threshold:
                self.neighbors.append(cell)

    def generate_new_position(self):
        theta = np.random.uniform(low=0, high=2 * np.pi)
        phi = np.random.uniform(low=0, high=np.pi)
        x = (self.radius * np.sin(phi) * np.cos(theta))
        y = (self.radius * np.sin(phi) * np.sin(theta))
        z = (self.radius * np.cos(phi))
        new_position = np.array([x, y, z])
        return new_position

    def reproduced(self):
        assert len(self.neighbors) < len(self.culture.cells)
        
        if self.available_space == True:    
            for attempt in range(self.max_repro_attempts):
                child_position = self.generate_new_position()
                # array with the distances from the proposed child position to the other cells
                distance = np.array([np.linalg.norm(child_position-cell.position) for cell in self.culture.cells])
                # boolean array specifying if there is no overlap between
                # the proposed child position and the other cells
                no_overlap = distance > 2 * self.radius
                # if it is true that there is no overlap for every element of the array, we break the loop
                if np.all(no_overlap):
                    break
            
            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available space
            if np.all(no_overlap):
                # we create a child in that position
                child_cell = Cell(position=child_position, culture=self.culture, adjacency_threshold=self.adjacency_threshold, radius=self.radius, max_repro_attempts=self.max_repro_attempts)
                # we add this cell to the culture's cells list 
                self.culture.cells.append(child_cell)
                # we find the child's neighbors
                child_cell.find_neighbors()
                # we add the child as a neighbor of its neighbors
                for cell in child_cell.neighbors:
                    cell.neighbors.append(child_cell)
                # we add the child to the graph (node and edges)
                self.culture.graph.add_node(child_cell)
                for cell in child_cell.neighbors:
                    self.culture.graph.add_edge(child_cell, cell)
            else:
                self.available_space = False
                #if there was no available space, we turn off reproduction
        else:
            pass
        # if the cell's neighbourhood is already full, we do nothing (reproduction is turned off)



class Culture:
    def __init__(self, adjacency_threshold=2, cell_radius=1, cell_max_repro_attempts=10000): # variables al pedo
        self.cell_max_repro_attempts=cell_max_repro_attempts
        self.adjacency_threshold=adjacency_threshold
        self.cell_radius=cell_radius
        first_cell=Cell(position=np.array([0, 0, 0]), culture=self, adjacency_threshold=self.adjacency_threshold, radius=self.cell_radius, max_repro_attempts=cell_max_repro_attempts)
        self.cells = [first_cell]
        self.graph = nx.Graph()
        self.graph.add_node(first_cell)
        
    def plot_culture(self):
        pass
    # to be implemented

    def plot_graph(self):
        pass
    # to be implemented

    def generate_adjacency_graph_from_scratch(self):
        self.graph=nx.Graph()
        for cell in self.cells:
            self.graph.add_node(cell)
        for i, cell1 in enumerate(self.cells):
            for cell2 in self.cells[i + 1 :]:
                if cell2 in cell1.neighbors:
                    self.graph.add_edge(cell1, cell2)


    def simulate(self, num_times):
        for i in range(num_times):
            np.random.shuffle(self.cells)
            for cell in self.cells:
                cell.reproduced()
