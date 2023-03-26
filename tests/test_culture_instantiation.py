import pytest

def test_single_cell_at_begining(request):
    sc = request.getfixturevalue("generic_cell_culture")
    assert len(sc.cells)==1