import numpy as np
from tumorsphere.core.cells import Cell


def test_cell_creation():
    """
    Test that a Cell object is created with the correct attributes.
    """
    culture = None  # replace with a Culture object
    position = np.array([0, 0, 0])
    is_stem = True
    parent_index = 0
    available_space = True
    creation_time = 0

    cell = Cell(
        position,
        culture,
        is_stem,
        parent_index,
        available_space,
        creation_time,
    )

    assert cell.culture == culture
    assert cell.is_stem == is_stem
    assert cell.parent_index == parent_index
    assert cell.neighbors_indexes == set()
    assert cell.available_space == available_space
    assert cell._index is not None


def test_cell_equality():
    """
    Test that two Cell objects are equal if they have the same attributes.
    """
    culture = None  # replace with a Culture object
    position = np.array([0, 0, 0])
    is_stem = True
    parent_index = 0
    available_space = True
    creation_time = 0

    cell1 = Cell(
        position,
        culture,
        is_stem,
        parent_index,
        available_space,
        creation_time,
    )
    cell2 = Cell(
        position,
        culture,
        is_stem,
        parent_index,
        available_space,
        creation_time,
    )

    assert cell1 == cell2


def test_cell_inequality():
    """
    Test that two Cell objects are not equal if they have different attributes.
    """
    culture = None  # replace with a Culture object
    position1 = np.array([0, 0, 0])
    position2 = np.array([1, 1, 1])
    is_stem = True
    parent_index = 0
    available_space = True
    creation_time = 0

    cell1 = Cell(
        position1,
        culture,
        is_stem,
        parent_index,
        available_space,
        creation_time,
    )
    cell2 = Cell(
        position2,
        culture,
        is_stem,
        parent_index,
        available_space,
        creation_time,
    )

    assert cell1 != cell2


# test for the following:
# if culture.conn is not None, test that database is written to correcly
# (the database should be a temporary file or deleted later)
# Give me the test below
# def test_cell_db_writing():
#     pass
