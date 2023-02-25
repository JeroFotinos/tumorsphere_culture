import numpy as np
import networkx as nx


class Cell:
    def __init__(self, pos, r):
        self.pos = pos
        self.r = r
        self.neighbors = []

    def reproduce(self):
        child_pos = None
        for n1 in self.neighbors:
            for n2 in n1.neighbors:
                if n2 != self and n2 not in self.neighbors:
                    # calculate the vector from self to n2
                    v = n2.pos - self.pos
                    # calculate the distance between self and n2
                    d = np.linalg.norm(v)
                    # calculate the distance between the centers of the spheres
                    c = d - self.r - n2.r
                    if c < 2 * self.r:
                        # if the distance between the centers of the spheres is less than 2*r,
                        # calculate the position of the new cell
                        child_pos = self.pos + (v / d) * (2 * self.r + c)
                        break
            if child_pos is not None:
                break

        if child_pos is not None:
            # create the new cell and add it to the culture
            child = Cell(child_pos, self.r)
            self.neighbors.append(child)
            child.neighbors.append(self)


class Culture:
    def __init__(self, r, a):
        self.r = r
        self.a = a
        self.cells = [Cell(np.array([0, 0, 0]), r)]
        self.graph = nx.Graph()
        self.graph.add_node(self.cells[0])

    def simulate(self, steps):
        for i in range(steps):
            # shuffle the cells to simulate random order
            np.random.shuffle(self.cells)
            for cell in self.cells:
                cell.reproduce()
                for n in cell.neighbors:
                    # calculate the distance between the centers of the spheres
                    d = np.linalg.norm(n.pos - cell.pos)
                    if d < self.a:
                        # if the distance between the centers of the spheres is less than a,
                        # add a link between the cells in the graph
                        self.graph.add_edge(cell, n)
