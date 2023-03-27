import copy

import numpy as np
import pytest

from tumorsphere.cells import Cell, Dcc
from tumorsphere.culture import Culture

# This file tests cell functions as well as culture instantiation and
# simulation. It does not test the ploting functions of the culture.

# ======= Instantiation and early stages =======


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
def test_single_cell_at_begining(cell_culture, request):
    """We test that there is only one cell when instantiating a culture."""
    single_cell_culture = request.getfixturevalue(cell_culture)
    assert len(single_cell_culture.cells) == 1


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [0, 1, 2, 3, 4])
def test_early_stage_cell_number(cell_culture, num_steps, request):
    """We test that every cell is reproducing in the early stages."""
    early_culture = request.getfixturevalue(cell_culture)
    early_culture.simulate(num_steps)
    assert len(early_culture.cells) == (2**num_steps)


@pytest.mark.parametrize(
    "culture, expected",
    [
        ("generic_cell_culture", Cell),
        ("dcc_seeded_culture", Dcc),
    ],
)
@pytest.mark.parametrize("num_steps", [0, 1, 2, 3, 4])
def test_cell_type_coherence(culture, expected, num_steps, request):
    """We test that Cell (Dcc) type seeded cultures
    yield cultures with only this type of cell.
    """
    cell_culture = request.getfixturevalue(culture)
    cell_culture.simulate(num_steps)
    type_coincidence = [
        isinstance(cell, expected) for cell in cell_culture.cells
    ]
    assert all(type_coincidence)


def test_initial_cell_type_error():
    """ValueError should be raised when trying to
    instantiate with an invlaid type of first cell.
    """
    with pytest.raises(ValueError):
        culture = Culture(first_cell_type="asd")


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
def test_neighbor_list_is_updating(cell_culture, request):
    """After one simulation step, there should be
    two cells, each the neighbor of the other.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(1)
    for cell in culture.cells:
        assert len(cell.neighbors) == 1
        # we assert that the 2 cells have 1 neighbor each


# ======= From scratch methods =======


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5])
def test_neighbors_from_scratch_matches_usual_function(
    cell_culture, num_steps, request
):
    """Test the find_neighbors_from_scratch() method of the Cell class.

    We check that erasing the neighbor of a cell from the culture, and
    then commanding that cell to find neighbors from scratch, returns a
    list of neighbors one element smaller.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    for cell in culture.cells:
        original_neighbors = cell.neighbors
        neighbor_to_remove = cell.neighbors[0]
        culture_with_one_cell_missing = copy.copy(culture)
        culture_with_one_cell_missing.cells = []
        for cell1 in culture.cells:
            if cell1 is not neighbor_to_remove:
                culture_with_one_cell_missing.cells.append(cell1)
        cell.culture = culture_with_one_cell_missing
        # we have to change the cell to the other culture, or it will
        # look for neighbors in the culture.cells list instead of the
        # culture_with_one_cell_missing.cells list
        cell.find_neighbors_from_scratch()
        assert len(cell.neighbors) == len(original_neighbors) - 1


# ======= Advanced stages =======

@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [8])
def test_max_number_of_neighbors(cell_culture, num_steps, request):
    """We test that the number of neighbors is less than the maximum allowed.

    The most compact packed culture is the one in which cells arrange themselves
    in a face-centered cubic (fcc), or equivalently, a hexagonal compact packaging (hcp).
    The maximum number of neighbors allowed is the sum of the first (12) and second (6)
    neighbors of a node in a fcc lattice.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    for cell in culture.cells:
        assert len(cell.neighbors) <= 18

@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [7])
def test_min_distance_between_cells(cell_culture, num_steps, request):
    """We test that minimum distance between cells is respected.

    (This tests assumes that all cells have the same radius.)
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    for cell1 in culture.cells:
        for cell2 in culture.cells:
            if cell2 is not cell1:
                assert (
                    np.linalg.norm(cell1.position - cell2.position)
                    >= 2 * cell1.radius
                )
