import numpy as np
import networkx as nx


class Cell:
    def __init__(self, position, culture, adjacency_threshold=2, radius=1):
        self.position = position
        # NumPy array, vector with 3 components
        self.culture = culture
        self.adjacency_threshold = adjacency_threshold
        # distance to second neighbor in HPC
        self.radius = radius  # radius of cell
        self.neighbors = []

    def find_neighbors(self):
        # self.neighbors = []
        # como las células no se mueven, sólo se pueden agregar vecinos, por
        # lo que no hay necesidad de reiniciar la lista
        for cell in self.culture.cells:
            if (cell is not self and cell not in self.neighbors) and np.linalg.norm(
                self.position - cell.position
            ) < self.adjacency_threshold:
                self.neighbors.append(cell)

    def reproduced(self):
        assert len(self.neighbors) < len(self.culture.cells)
        
        for attempt in range(max_attempts): #definir max attempts
            theta = np.random.uniform(0, 2 * math.pi) #chequear
            phi = np.random.uniform(0, math.pi) #cambiar pi por el de numpy
            x = (self.radius * math.sin(phi) * math.cos(theta))
            y = (self.radius * math.sin(phi) * math.sin(theta)) #cambiar senos y consenos por los de numpy
            z = (self.radius * math.cos(phi))
            position = (x, y, z)
        
        # Check if the new sphere would overlap with any existing spheres
        overlap = False
        for cell in self.culture.cells:
            distance = np.linalg.norm(
                self.position - cell.position
            ) math.sqrt((sphere[0] - x)**2 + (sphere[1] - y)**2 + (sphere[2] - z)**2)
            if distance < min_distance:
                overlap = True
                break
        
        # If there is no overlap, return the new sphere's position
        if not overlap:
            return position

        # ===========================
         
    def reproduced2(self):
        if len(self.neighbors) < len(self.culture.cells):
            while True:
                offset = np.random.uniform(-self.r, self.r, size=3)
                new_pos = self.position + 2*self.r * (self.position - self.culture.center) / np.linalg.norm(self.position - self.culture.center) + offset
                if np.all([np.linalg.norm(new_pos - cell.position) >= 2*self.r for cell in self.neighbors]):
                    break
            new_cell = Cell(new_pos, self.culture, self.a, self.r)
            new_cell.find_neighbors()
            self.culture.cells.append(new_cell)
            self.neighbors.append(new_cell)
            self.culture.graph.add_node(new_cell)
            self.culture.graph.add_edge(self, new_cell)
            for cell in new_cell.neighbors:
                cell.neighbors.append(new_cell)
            new_cell.reproduced()
            return new_cell



class Culture:
    def __init__(self, num_cells, a, r):
        self.cells = [Cell(np.array([0, 0, 0]))]
        self.graph = nx.Graph()
        self.center = np.mean([cell.position for cell in self.cells], axis=0)
        for cell in self.cells:
            cell.find_neighbors()
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
