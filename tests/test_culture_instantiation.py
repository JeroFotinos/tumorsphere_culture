import copy

import pytest

from tumorsphere.cells import Cell, Dcc
from tumorsphere.culture import Culture


# ======= Instantiation and early stages =======


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
def test_single_cell_at_begining(cell_culture, request):
    single_cell_culture = request.getfixturevalue(cell_culture)
    assert len(single_cell_culture.cells) == 1


@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [0, 1, 2, 3, 4])
def test_early_stage_cell_number(cell_culture, num_steps, request):
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
    cell_culture = request.getfixturevalue(culture)
    cell_culture.simulate(num_steps)
    type_coincidence = [
        isinstance(cell, expected) for cell in cell_culture.cells
    ]
    assert all(type_coincidence)


def test_initial_cell_type_error():
    with pytest.raises(ValueError):
        culture = Culture(first_cell_type='asd')

@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
def test_neighbor_list_is_updating(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(1)
    for cell in culture.cells:
        assert len(cell.neighbors) == 1
        # we assert that the 2 cells have 1 neighbor each



# ======= From scratch methods =======

@pytest.mark.xfail(reason="not gonna work before test_neighbor_list_is_updating passes")
@pytest.mark.parametrize(
    "cell_culture",
    ["generic_cell_culture", "dcc_seeded_culture", "csc_seeded_culture"],
)
@pytest.mark.parametrize("num_steps", [1, 2, 3, 4, 5])
def test_neighbors_from_scratch_matches_usual_function(cell_culture, num_steps, request):
    culture = request.getfixturevalue(cell_culture)
    culture.simulate(num_steps)
    for cell in culture.cells:
        original_neighbors = copy.deepcopy(cell.neighbors)
        culture_with_one_cell_missing = copy.deepcopy(culture)
        culture_with_one_cell_missing.cells.remove(cell.neighbors[0])
        cell.find_neighbors_from_scratch()
        assert len(cell.neighbors) == len(original_neighbors)-1
        