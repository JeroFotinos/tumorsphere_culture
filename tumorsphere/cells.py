import copy
import random

import matplotlib.pyplot as plt
import networkx as nx
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

    def find_neighbors(self):
        # first we construct the list of neighbors and neighbors of neighbors
        neighbors_and_neigh_of_neigh = set(self.neighbors)
        # if the cell is a newborn, it will only have its parent as neighbor,
        # so neighbors of its neighbors are just the neighbors of its parent.
        # The first time we have to go a level deeper.
        if len(self.neighbors) < 20:
            for cell1 in self.neighbors:
                neighbors_and_neigh_of_neigh = (
                    neighbors_and_neigh_of_neigh.union(set(cell1.neighbors))
                )
                for cell2 in cell1.neighbors:
                    neighbors_and_neigh_of_neigh = (
                        neighbors_and_neigh_of_neigh.union(
                            set(cell2.neighbors)
                        )
                    )
                    for cell3 in cell2.neighbors:
                        neighbors_and_neigh_of_neigh = (
                            neighbors_and_neigh_of_neigh.union(
                            set(cell3.neighbors)
                            )
                        )
            neighbors_and_neigh_of_neigh = list(neighbors_and_neigh_of_neigh)
        else:
            for cell1 in self.neighbors:
                neighbors_and_neigh_of_neigh = (
                    neighbors_and_neigh_of_neigh.union(set(cell.neighbors))
                )
                for cell2 in cell1.neighbors:
                    neighbors_and_neigh_of_neigh = (
                        neighbors_and_neigh_of_neigh.union(
                            set(cell2.neighbors)
                        )
                    )
            neighbors_and_neigh_of_neigh = list(neighbors_and_neigh_of_neigh)
        # now we check if there are cells to append
        for cell in neighbors_and_neigh_of_neigh:
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
        # first we construct the list of neighbors and neighbors of neighbors
        neighbors_and_neigh_of_neigh = set(self.neighbors)
        # if the cell is a newborn, it will only have its parent as neighbor,
        # so neighbors of its neighbors are just the neighbors of its parent.
        # The first time we have to go a level deeper.
        if len(self.neighbors) < 20:
            for cell1 in self.neighbors:
                neighbors_and_neigh_of_neigh = (
                    neighbors_and_neigh_of_neigh.union(set(cell1.neighbors))
                )
                for cell2 in cell1.neighbors:
                    neighbors_and_neigh_of_neigh = (
                        neighbors_and_neigh_of_neigh.union(
                            set(cell2.neighbors)
                        )
                    )
                    for cell3 in cell2.neighbors:
                        neighbors_and_neigh_of_neigh = (
                            neighbors_and_neigh_of_neigh.union(
                            set(cell3.neighbors)
                            )
                        )
            neighbors_and_neigh_of_neigh = list(neighbors_and_neigh_of_neigh)
        else:
            for cell1 in self.neighbors:
                neighbors_and_neigh_of_neigh = (
                    neighbors_and_neigh_of_neigh.union(set(cell.neighbors))
                )
                for cell2 in cell1.neighbors:
                    neighbors_and_neigh_of_neigh = (
                        neighbors_and_neigh_of_neigh.union(
                            set(cell2.neighbors)
                        )
                    )
            neighbors_and_neigh_of_neigh = list(neighbors_and_neigh_of_neigh)
        # we reset the neighbors list
        self.neighbors = []
        # we add the cells to the list
        for cell in neighbors_and_neigh_of_neigh:
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
                if self.is_stem:
                    if np.random.uniform() <= self.prob_stem:
                        child_cell = Cell(
                            position=child_position,
                            culture=self.culture,
                            adjacency_threshold=self.adjacency_threshold,
                            radius=self.radius,
                            is_stem=True,
                            max_repro_attempts=self.max_repro_attempts,
                            prob_stem=self.prob_stem,
                            continuous_graph_generation=self._continuous_graph_generation,
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
                        )
                        if np.random.uniform() <= self._swap_probability:
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
                    )
                # we add this cell to the culture's cells list
                self.culture.cells.append(child_cell)
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
                # if there was no available space, we turn off reproduction
        # else:
        #     pass
        # if the cell's neighbourhood is already full, we do nothing (reproduction is turned off)
