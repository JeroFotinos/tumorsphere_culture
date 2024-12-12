"""
Module containing the Cell class used for simulating cells in a culture.

Classes:
    - Cell: Represents a single cell in a culture. Dependent on the Culture
    class.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING, Dict

import numpy as np

if TYPE_CHECKING:
    from tumorsphere.core.culture import Culture
# We avoid the circular import at runtime by using TYPE_CHECKING to
# conditionally import Culture only when type checking. Then, we use
# forward references (i.e., "Culture") in the type hints for the culture
# attribute and __init__ parameters.


@dataclass(frozen=False, slots=True)
class Cell:
    """Dataclass that represents a single cell in a Culture.

    Attributes
    ----------
    culture: Culture
        The culture to which the cell belongs.
    is_stem: bool
        Whether the cell is a stem cell or not.
    aspect_ratio: float
        The ratio of the cells width to its height.
    parent_index: Optional[int]
        The index of the parent cell in the culture's cell_positions array.
        Default is 0.
    available_space: bool
        Whether the cell has available space around it or not. Default is True.
    _index: Optional[int]
        The index of the cell's position in the culture's cell_positions array.
        It's not directly settable during instantiation.

    Methods
    -------
    __init__(
        position, culture, is_stem, phi=None, aspect_ratio=1, parent_index=0,
        parent_index=0, available_space=True
        )
        Initializes the Cell object and sets the _index attribute
        based on the position given.

    Notes
    -----
    Since slots are used in this dataclass, multiple inheritance is not
    supported.

    """

    culture: "Culture"
    is_stem: bool
    aspect_ratio: float = 1
    parent_index: Optional[int] = 0
    available_space: bool = True
    _index: Optional[int] = field(default=False, init=False)
    shrink: bool = False
    neighbors_relative_pos: Dict[int, np.ndarray] =  field(default_factory=dict)
    neighbors_overlap: Dict[int, float] =  field(default_factory=dict)

    def __init__(
        self,
        position: np.ndarray,
        culture: "Culture",
        is_stem: bool,
        phi: float = None,
        aspect_ratio: float = 1,
        parent_index: Optional[int] = 0,
        shrink: bool = False,
        available_space: bool = True,  # not to be set by user
        creation_time: int = 0,
    ) -> None:
        """
        Initializes the Cell object.

        Parameters
        ----------
        position : np.ndarray
            The position of the cell. This is used to update the
            cell_positions in the culture and set the _index
            attribute, but is not stored as an attribute in the object itself.
        culture : Culture
            The culture to which the cell belongs.
        is_stem : bool
            Whether the cell is a stem cell or not.
        phi : float
            The angle in the x-y plane of the cell. This is used to update the
            cell_phies in the culture.
        aspect_ratio: float
            The ratio of the cells width to its height.
        parent_index : Optional[int], default=0
            The index of the parent cell in the culture's cell_positions
            array.
        neighbors_relative_pos : Dict[int, np.ndarray]
            A dictionary where the keys are the indices of the neighbors, and the values 
            are their relative positions with respect to the reference cell.
        neighbors_overlap : Dict[int, float]
            A dictionary where the keys are the indices of the neighbors, and the values 
            are their overlap with the reference cell.
        available_space : bool, default=True
            Whether the cell has available space around it or not.
        shrink : bool, default=False
            Whether the cell has to shrink or not.

        Notes
        ------
        Having the aspect ratio of the cell and if every cell has the same area,
        we can calculate the semi major axis (l_par) and semi minor axis (l_perp)
        with:
        l_par = np.sqrt((cell_area*cell.aspect_ratio)/np.pi)
        l_perp = np.sqrt(cell_area/(np.pi*cell.aspect_ratio))

        """
        self.culture = culture
        self.is_stem = is_stem
        self.parent_index = parent_index
        self.available_space = available_space
        self.aspect_ratio = aspect_ratio
        self.shrink = shrink

        # we initialize the dictionary for storing neighbors' relative 
        # positions
        self.neighbors_relative_pos = dict()
        # and the overlap with the neighbors
        self.neighbors_overlap = dict()
        # we FIRST get the cell's index
        self._index = len(culture.cell_positions)

        # and THEN add the cell to the culture's position matrix and cell
        # lists, in the previous index
        culture.cell_positions = np.append(
            culture.cell_positions, [position], axis=0
        )
        self.culture.cells.append(self)
        self.culture.active_cell_indexes.append(self._index)

        # and add the cell to the culture's phi matrix
        culture.cell_phies = np.append(culture.cell_phies, phi)

        # We also add the cell to the culture's spatial hash grid
        self.culture.grid.add_cell_to_hash_table(
            self._index,
            position,
        )

        # if we're simulating with the SQLite DB, we insert a register in the
        # Cells table of the SQLite DB
        self.culture.output.record_cell(
            self._index,
            int(self.parent_index),
            self.culture.cell_positions[self._index][0],
            self.culture.cell_positions[self._index][1],
            self.culture.cell_positions[self._index][2],
            creation_time,
            self.is_stem,
        )

    def velocity(self):
        """
        It returns the velocity vector of the given cell.

        Returns
        -------
        np.ndarray
            The velocity vector of the cell.
        """
        if np.isclose(self.aspect_ratio, 1):
            speed = 0
        else:
            speed = 1
        return speed * np.array(
            [
                np.cos(self.culture.cell_phies[self._index]),
                np.sin(self.culture.cell_phies[self._index]),
                0,
            ]
        )

