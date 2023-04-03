import copy

import numpy as np
import pytest

from tumorsphere.cells import Cell
from tumorsphere.culture import Culture

# This file tests cell functions as well as culture instantiation and
# simulation. It does not test the ploting functions of the culture.

# ======= Instantiation and early stages =======


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_single_cell_at_begining(cell_culture, request):
    """We test that there is only one cell when instantiating a culture."""
    single_cell_culture = request.getfixturevalue(cell_culture)
    assert len(single_cell_culture.cells) == 1


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [0, 1, 2, 3, 4])
def test_early_stage_cell_number(cell_culture, num_steps, request):
    """We test that every cell is reproducing in the early stages."""
    early_culture = request.getfixturevalue(cell_culture)
    early_culture.simulate(num_steps)
    assert len(early_culture.cells) == (2**num_steps)


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [0, 1, 2, 3, 4])
def test_cell_type_coherence(cell_culture, num_steps, request):
    """Differentiated cells only give differentiated cells.

    We test that cultures seeded with DCCs (first_cell_is_stem=False)
    yield cultures were all cells have is_stem = Flase.
    """
    cell_culture = request.getfixturevalue(cell_culture)
    cell_culture.simulate(num_steps)
    type_coincidence = [(cell.is_stem is False) for cell in cell_culture.cells]
    assert all(type_coincidence)


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
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


@pytest.mark.skip(reason="need to rethink how to do the test")
@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [5, 6, 7])
@pytest.mark.parametrize("cell_number", [0, 1, 2, 3, 4, 5])
def test_neighbors_from_scratch_matches_usual_function_new(
    cell_culture, num_steps, cell_number, request
):
    """Test the find_neighbors_from_scratch() method of the Cell class.

    We check that erasing the first neighbor of the first cell from the
    culture, and then commanding that cell to find neighbors from scratch,
    returns a list of neighbors one element smaller than the original.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    cell = culture.cells[cell_number]
    original_number_of_neighbors = len(cell.neighbors)
    neighbor = cell.neighbors[0]
    for cell in neighbor.neighbors:
        cell.neighbors.remove(neighbor)
    del neighbor
    # this may change the cell_number of cell in cell.culture.cells
    cell.find_neighbors_from_entire_culture_from_scratch()
    assert len(cell.neighbors) == original_number_of_neighbors - 1


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [5, 6])
@pytest.mark.parametrize("cell_number", [0, 1, 2, 3, 4, 5])
def test_neighbors_match_neighbors_from_entire_culture_from_scratch(
    cell_culture, num_steps, cell_number, request
):
    """Test the find_neighbors() method of the Cell class.

    We check that the adjacency_threshold is appropriate for reproducing
    the neighbors list one would obtain by looking for neighboring relations
    with every cell of the culture, which can be done by using the
    find_neighbors_from_entire_culture_from_scratch() method.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    cell = culture.cells[cell_number]
    original_number_of_neighbors = len(cell.neighbors)
    cell.find_neighbors_from_entire_culture_from_scratch()
    assert len(cell.neighbors) == original_number_of_neighbors


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [8])
def test_neighbors_match_neighbors_from_entire_culture_from_scratch_exactly(
    cell_culture, num_steps, request
):
    """Test the find_neighbors_from_scratch() method of the Cell class.

    Similar to test_neighbors_match_neighbors_from_entire_culture_from_scratch,
    but we check that the lists match exactly, i.e. for every cell.
    """
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    for cell in culture.cells:
        original_number_of_neighbors = len(cell.neighbors)
        cell.find_neighbors_from_entire_culture_from_scratch()
        assert len(cell.neighbors) == original_number_of_neighbors

# ======= Intermediate stages =======


@pytest.mark.skip(
    reason="I augmented the adjacency threshold beyond the second neighbor distance of the hcp"
)
@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [8])
def test_max_number_of_neighbors(cell_culture, num_steps, request):
    """We test that the number of neighbors is less than the maximum allowed.

    The most compactly packed culture is the one in which cells arrange themselves
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
    ["dcc_seeded_culture", "csc_seeded_culture"],
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


# ======= Swap between CSC and its DCC child =======


def test_stemness_swapping_between_csc_and_dcc_child(request):
    """We test if the swapping of positions is taking place.

    To test if the swap is working, we check if, when _swap_probability = 1,
    and the csc is forced to give only dcc childs (i.e. prob_stem = 0), after
    one simulation step its original position is given to the dcc child.
    """
    culture = request.getfixturevalue("csc_seeded_culture")
    culture.cells[0]._swap_probability = 1
    culture.cells[0].prob_stem = 0
    culture.simulate(1)
    assert culture.cells[1].is_stem
    assert not culture.cells[0].is_stem
