import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

colors = {"cell": "green", "csc": "red", "dcc": "blue"}

# probabilities = {'ps' : 0.36, 'pd' : 0.16}
# prob_stem = 0.36


class Cell:
    def __init__(
        self,
        position,
        culture,
        adjacency_threshold=2 * np.sqrt(2),
        radius=1,
        max_repro_attempts=10000,
        continuous_graph_generation=False,
    ):
        self.position = position
        # NumPy array, vector with 3 components
        self.culture = culture
        self.adjacency_threshold = adjacency_threshold
        # distance to second neighbor in HPC
        self.radius = radius  # radius of cell
        self.max_repro_attempts = max_repro_attempts
        self._continuous_graph_generation = continuous_graph_generation
        self.color = colors["cell"]
        self.neighbors = []
        self.available_space = True

    # TO DO: a find_neighbors_from_neighbors method that
    # looks for neighbors of a cell in their neighbors and
    # their neighbor's neighbors. This will speed things up.
    # We can implement a test that checks that the result is
    # the same.

    def find_neighbors_from_scratch(self):
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

    def find_neighbors(self):
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
                # array with the distances from the proposed child position to the other cells
                distance = np.array(
                    [
                        np.linalg.norm(child_position - cell.position)
                        for cell in self.culture.cells
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
                child_cell = Cell(
                    position=child_position,
                    culture=self.culture,
                    adjacency_threshold=self.adjacency_threshold,
                    radius=self.radius,
                    max_repro_attempts=self.max_repro_attempts,
                    continuous_graph_generation=self._continuous_graph_generation,
                )
                # we add this cell to the culture's cells list
                self.culture.cells.append(child_cell)
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
                # if there was no available space, we turn off reproduction
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing (reproduction is turned off)


class Dcc(Cell):
    def __init__(
        self,
        position,
        culture,
        adjacency_threshold=2 * np.sqrt(2),
        radius=1,
        max_repro_attempts=10000,
        continuous_graph_generation=False,
    ):
        super().__init__(
            position,
            culture,
            adjacency_threshold,
            radius,
            max_repro_attempts,
            continuous_graph_generation,
        )
        self.color = colors["dcc"]

    def reproduce(self):
        assert len(self.neighbors) <= len(self.culture.cells)

        if self.available_space:
            for attempt in range(self.max_repro_attempts):
                child_position = self.generate_new_position()
                # array with the distances from the proposed child position to the other cells
                distance = np.array(
                    [
                        np.linalg.norm(child_position - cell.position)
                        for cell in self.culture.cells
                    ]
                )
                # boolean array specifying if there is no overlap between
                # the proposed child position and the other cells
                no_overlap = np.all(
                    distance >= 2 * self.radius
                )  # or np.all(np.allclose(distance, self.radius)) ###### ACÁ HAY ALGO
                # if it is true that there is no overlap for every element of the array, we break the loop
                if no_overlap:
                    break

            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available space
            if no_overlap:
                # we create a child in that position
                child_cell = Dcc(
                    position=child_position,
                    culture=self.culture,
                    adjacency_threshold=self.adjacency_threshold,
                    radius=self.radius,
                    max_repro_attempts=self.max_repro_attempts,
                    continuous_graph_generation=self._continuous_graph_generation,
                )
                # we add this cell to the culture's cells list
                self.culture.cells.append(child_cell)
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
                # if there was no available space, we turn off reproduction


class Csc(Cell):
    def __init__(
        self,
        position,
        culture,
        adjacency_threshold=2 * np.sqrt(2),
        radius=1,
        max_repro_attempts=10000,
        prob_stem=0.36,
        continuous_graph_generation=False,
    ):
        super().__init__(
            position,
            culture,
            adjacency_threshold,
            radius,
            max_repro_attempts,
            continuous_graph_generation,
        )
        self.color = colors["csc"]
        self.prob_stem = prob_stem
        self._swap_probability = 0.5

    def reproduce(self):
        assert len(self.neighbors) <= len(self.culture.cells)

        if self.available_space:
            for attempt in range(self.max_repro_attempts):
                child_position = self.generate_new_position()
                # array with the distances from the proposed child position to the other cells
                distance = np.array(
                    [
                        np.linalg.norm(child_position - cell.position)
                        for cell in self.culture.cells
                    ]
                )
                # boolean specifying if there is no overlap between the
                # proposed child position and any of the other cells
                no_overlap = np.all(distance >= 2 * self.radius)
                # if it is true that there is no overlap for every element of the array, we break the loop
                if no_overlap:
                    break

            # if there was no overlap, we create a child in that position
            # if not, we do nothing but specifying that there is no available space
            if no_overlap:
                # we create a child in that position
                # a csc with prob prob_stem and a dcc if not
                if np.random.uniform() <= self.prob_stem:
                    child_cell = Csc(
                        position=child_position,
                        culture=self.culture,
                        adjacency_threshold=self.adjacency_threshold,
                        radius=self.radius,
                        max_repro_attempts=self.max_repro_attempts,
                        prob_stem=self.prob_stem,
                        continuous_graph_generation=self._continuous_graph_generation,
                    )
                    # we add this cell to the culture's cells list
                    self.culture.cells.append(child_cell)
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
                    child_cell = Dcc(
                        position=child_position,
                        culture=self.culture,
                        adjacency_threshold=self.adjacency_threshold,
                        radius=self.radius,
                        max_repro_attempts=self.max_repro_attempts,
                        continuous_graph_generation=self._continuous_graph_generation,
                    )
                    # we add this cell to the culture's cells list
                    self.culture.cells.append(child_cell)
                    # we find the child's neighbors
                    child_cell.find_neighbors()
                    # we swap the positions and neighbors with probability 1/2
                    if np.random.uniform() <= self._swap_probability:
                        self.swap_self_with_dcc_child_and_arrange_neighbors(
                            child=child_cell
                        )
                    else:
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
                # if there was no available space, we turn off reproduction

    def swap_self_with_dcc_child_and_arrange_neighbors(self, child):
        # we copy the parent's original information
        old_parent_position = copy.copy(self.position)
        old_parent_neighbors = copy.copy(self.neighbors)
        # we remove self from the neighbors of its neighbors
        # and add child_cell instead
        for cell in self.neighbors:
            cell.neighbors.remove(self)
            cell.neighbors.append(child)
        # we assign the info of the child to the parent
        self.position = copy.copy(child.position)
        self.neighbors = copy.copy(child.neighbors)
        # we add self to the neighbors of the neighbor cells found
        # by the child when it was in the old position (if it is
        # not already there)
        for cell in child.neighbors:
            if self not in cell.neighbors:
                cell.neighbors.append(self)
        # we assign the child, the old info of the parent
        child.position = old_parent_position
        child.neighbors = old_parent_neighbors
        # we update the graph
        if self._continuous_graph_generation == True:
            # we remove the old node and its edges from the graph
            self.culture.graph.remove_node(self)
            # we add the node again with its new edges
            self.culture.graph.add_node(self)
            for cell in self.neighbors:
                self.culture.graph.add_edge(self, cell)
            # we add the child node and its edges
            self.culture.graph.add_node(child)
            for cell in child.neighbors:
                self.culture.graph.add_edge(child, cell)
