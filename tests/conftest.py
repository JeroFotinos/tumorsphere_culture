import pytest

from tumorsphere.culture import Culture


@pytest.fixture()
def csc_seeded_culture():
    csc_culture = Culture(cell_max_repro_attempts=500, first_cell_is_stem=True)
    return csc_culture


@pytest.fixture()
def dcc_seeded_culture():
    dcc_culture = Culture(cell_max_repro_attempts=500)
    return dcc_culture
