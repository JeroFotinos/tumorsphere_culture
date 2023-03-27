import pytest

from tumorsphere.culture import Culture


@pytest.fixture()
def generic_cell_culture():
    culture = Culture(cell_max_repro_attempts=500)
    return culture


@pytest.fixture()
def csc_seeded_culture():
    csc_culture = Culture(cell_max_repro_attempts=500, first_cell_type="csc")
    return csc_culture


@pytest.fixture()
def dcc_seeded_culture():
    dcc_culture = Culture(cell_max_repro_attempts=500, first_cell_type="dcc")
    return dcc_culture
