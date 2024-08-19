"""
Module containing the Cell class used for simulating cells in a culture.

Classes:
    - Cell: Represents a single cell in a culture. Dependent on the Culture
    class.
"""

from dataclasses import dataclass, field
from typing import Optional, Set

import numpy as np


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
    neighbors_indexes: Set[int]
        A set of indexes corresponding to the neighboring cells in the
        culture's cell_positions array. Default is an empty set.
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
    neighbors_indexes: Set[int] = field(default_factory=set)
    available_space: bool = True
    _index: Optional[int] = field(default=False, init=False)

    def __init__(
        self,
        position: np.ndarray,
        culture: "Culture",
        is_stem: bool,
        phi: float = None,
        aspect_ratio: float = 1,
        parent_index: Optional[int] = 0,
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
        neighbors_indexes : Set[int], default=set()
            A set of indexes corresponding to the neighboring cells in the
            culture's cell_positions array.
        available_space : bool, default=True
            Whether the cell has available space around it or not.

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
        self.neighbors_indexes = set()
        self.available_space = available_space
        self.aspect_ratio = aspect_ratio

        # we FIRST get the cell's index
        self._index = len(culture.cell_positions)

        # and THEN add the cell to the culture's position matrix,
        # to the angle matrix and cell lists, in the previous index
        culture.cell_positions = np.append(
            culture.cell_positions, [position], axis=0
        )

        culture.cell_phies = np.append(culture.cell_phies, phi)

        self.culture.cells.append(self)
        self.culture.active_cell_indexes.append(self._index)

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

    # ---------------------------------------------------------
    def velocity(self):
        """
        It returns the velocity vector of the given cell  taking into account the 
        mobility.

        Returns
        -------
        np.ndarray
            The velocity vector of the cell.
        """

        # longitudinal & transversal mobility
        if self.aspect_ratio != 1:  # np.close?
            mP = (
                1
                / np.sqrt((self.culture.cell_area * self.aspect_ratio) / np.pi)
                * (3 * self.aspect_ratio / 4.0)
                * (
                    (self.aspect_ratio) / (1 - self.aspect_ratio**2)
                    + (2.0 * self.aspect_ratio**2 - 1.0)
                    / np.power(self.aspect_ratio**2 - 1.0, 1.5)
                    * np.log(
                        self.aspect_ratio + np.sqrt(self.aspect_ratio**2 - 1.0)
                    )
                )
            )
        else:
            mP = 1 / np.sqrt(
                (self.culture.cell_area * self.aspect_ratio) / np.pi
            )

        # we suppose kProp linear with s_epsA
        s_epsA = (self.aspect_ratio**2 - 1) / (self.aspect_ratio**2 + 1)
        # we use the constant in order to have kProp=3 for aspect_ratio=5
        kProp = 13/4*s_epsA
        s_v0 = kProp * mP 
        # kProp=3 o s_v0=3?????

        return s_v0 * np.array(
            [
                np.cos(self.culture.cell_phies[self._index]),
                np.sin(self.culture.cell_phies[self._index]),
                0,
            ]
        )