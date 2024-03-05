import numpy as np

import pytest


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_first_cell_has_index_0(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    assert culture.cells[0]._index == 0


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_first_cell_has_space(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    assert culture.cells[0].available_space is True


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_first_position_correctly_added_to_matrix(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    assert culture.cell_positions[0].all() == np.array([0, 0, 0]).all()


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_cell_added_to_cells_list(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    assert len(culture.cells) == 1


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_cell_added_to_active_cells_list(cell_culture, request):
    culture = request.getfixturevalue(cell_culture)
    assert len(culture.active_cell_indexes) == 1


@pytest.mark.skip("Not implemented yet")
def test_database_cells_record():
    pass


@pytest.mark.skip("Not implemented yet")
def test_database_StemChanges_record():
    pass
