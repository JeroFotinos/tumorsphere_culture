"""Module that defines the SpatialHashGrid class."""

from itertools import product
from typing import Tuple
from collections import defaultdict

import numpy as np

# from line_profiler import profile


class SpatialHashGrid:
    def __init__(
        self,
        culture=None,  # type Culture, not declared to avoid circular imports
        bounds: float = None,
        cube_size: float = 2,
        torus: bool = True,
    ) -> None:
        """
        Initialize the Spatial Hashing of a Culture.

        Parameters
        ----------
        culture : Culture
            The Culture to be spatially hashed. None at initialization, but
            meant to be set by `tumorsphere.core.culture.Culture.__init__()`.
        bounds : int, optional
            The bounds of the grid, by default None. If None, the space is
            unbouded. If provided, the space is bounded to the [0, bounds)^3
            cube.
        cube_size : int, optional
            The size of the cubes in the grid, by default 2. This value comes
            from considering that cells have usually radius 1, so a cube of
            side $h=2r$ is enough to make sure that we only have to check
            superpositions with cells on the same or first neighboring grid
            cells. Enlarge if using larger cells.
        torus : bool, optional
            Whether the grid is a torus or not, only relevant when bounds are
            provided, True by default. If True, the grid is a torus, so the
            cells that go out of the bounds appear on the other side of the
            grid. If False, the grid is a bounded cube, so behavior should be
            defined to manage what happens when cells go out of the bounds of
            the simulation.

        Attributes
        ----------
        hash_table : Dict[Tuple[int, int, int], Set[int]]
            A dictionary that maps the coordinates of the grid to the set of
            cell indices that are in that cube.
        """
        self.culture = culture
        self.torus = torus
        self.bounds = bounds
        self.cube_size = cube_size
        self.offsets = np.array(list(product(range(-1, 2), repeat=3)))
        self.hash_table = defaultdict(set)

    def get_hash_key(
        self,
        position: np.ndarray,
    ) -> Tuple[int, int, int]:
        """Get the hash key of a position."""
        return tuple(np.floor(position / self.cube_size).astype(int))

    def add_cell_to_hash_table(
        self,
        cell_index: int,
        position: np.ndarray,
    ) -> None:
        """Add a cell to the hash table."""
        self.hash_table[self.get_hash_key(position)].add(cell_index)

    def remove_cell_from_hash_table(
        self,
        cell_index: int,
        position: np.ndarray,
    ) -> None:
        """Remove a cell from the hash table."""
        self.hash_table[self.get_hash_key(position)].remove(cell_index)

    def is_position_in_bounds(
        self,
        position: np.ndarray,
    ) -> bool:
        """Check if a position is in the bounds of the grid."""
        if self.bounds is None:
            return True
        return np.all(position >= 0) and np.all(position < self.bounds)

    def get_in_bounds_position(
        self,
        position: np.ndarray,
    ) -> np.ndarray:
        """Get the updated position within the bounds of the grid, given
        an old, out-of-bounds position.

        For a toroidal grid, positions wrap around the edges to the
        opposite side. For a bounded, non-toroidal grid, this method raises
        a NotImplementedError.

        Parameters
        ----------
        position : np.ndarray
            The original position of the cell.

        Returns
        -------
        np.ndarray
            The adjusted position within the bounds of the grid.

        Raises
        ------
        NotImplementedError
            If the grid is bounded and not toroidal.
        """
        if self.bounds is None:
            return position

        if self.torus:
            # Use modulus operation to wrap around for a toroidal grid.
            # This handles both positive and negative positions, ensuring
            # they wrap around correctly.
            return np.mod(position, self.bounds)
        else:
            # If the grid is not a torus and has defined bounds, raise a
            # NotImplementedError. This informs the user that behavior for
            # bounded, non-torus grids needs to be defined. I didn't
            # implement this because we could handle this case in a number
            # of ways: we could kill the cell, we could bounce it back,
            # we could reject the move/reproduction attempt, etc.
            raise NotImplementedError(
                "Non-torus grid boundary behavior is not yet implemented."
            )

    # @profile
    def find_neighbors(self, position: np.ndarray) -> set:
        """Returns the set of indexes of all existing cells within a 3D Moore
        neighborhood of a given position.

        This method considers cells in the same and adjacent cubes as
        neighbors. With this, the set of neighbors is the set of indexes of
        existing cells that would neighbor a new cell in the provided
        position. Note that acording to this, a cell is always a neighbor of
        itself.

        Parameters
        ----------
        position : np.ndarray
            The position of the target cell in the grid.

        Returns
        -------
        set
            A set of cell identifiers that are considered neighbors of a new
            cell in the provided position.
        """
        neighbors = set()
        # Convert key to array once
        key = np.array(self.get_hash_key(position))

        # Broadcasting addition to get adjacent keys
        adj_keys = key + self.offsets

        # Handle toroidal wrapping
        if self.bounds is not None and self.torus:
            adj_keys = np.mod(adj_keys, self.bounds)

        # Convert array of keys to tuples and filter valid keys
        for adj_key in map(tuple, adj_keys):  # Convert once and iterate
            neighbors.update(self.hash_table[adj_key])

        return neighbors
