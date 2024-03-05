import os

import numpy as np

import pytest

from tumorsphere.core.culture import Culture


@pytest.mark.parametrize(
    "cell_culture",
    ["dcc_seeded_culture", "csc_seeded_culture"],
)
def test_single_cell_at_begining(cell_culture, request):
    """We test that there is only one cell when instantiating a culture."""
    single_cell_culture = request.getfixturevalue(cell_culture)
    assert len(single_cell_culture.cells) == 1


@pytest.mark.parametrize(
    "evolved_culture, expected_number_of_cells",
    [
        ("culture_evolved_at_time_2", "2"),
        ("culture_evolved_at_time_3", "4"),
    ],
)
def test_early_stage_cell_number(
    evolved_culture, expected_number_of_cells, request
):
    """We test that every cell is reproducing in the early stages, for t=2 and
    t=3.

    Note that since the loop in Culture.simulate() is range(1, num_times), we
    need to pass num_times=3 to simulate 2 timesteps, i.e.
    - `num_times` is the time t, starts at 1 and it's used as (e.g.) creation time;
    - `num_times - 1` is the number of duplications (timesteps).

    Therefore, we expect 2**(num_times - 1) cells.
    """
    # Create and simulate the culture
    culture = request.getfixturevalue(evolved_culture)

    # Perform your assertions
    assert len(culture.cells) == int(expected_number_of_cells)


# # @pytest.mark.skip("Not implemented yet")
# @pytest.mark.parametrize(
#     "culture", ["deactivated_culture"]
# )
# def test_deactivation(culture, request):
#     """We test that all cells are deactivated after trying to reproduce with
#     no space available (but previously having space available, so that they
#     have self.available_space = True)."""
#     # we get the culture
#     culture = request.getfixturevalue(culture)

#     # we assert that all cells were deactivated
#     # assert all([cell.available_space is False for cell in culture.cells])

#     # we assert that the active cells list is empty
#     # assert len(culture.active_cell_indexes) == 0

#     # we identify the first cell
#     first_cell = culture.cells[0]

#     # we assert that it's located at the origin
#     # assert first_cell.position.all() == np.array([0, 0, 0]).all()

#     # we reactivate the first cell (which at this time, t=6, does not have enough
#     # space around it to reproduce)
#     first_cell.available_space = True
#     culture.active_cell_indexes.append(first_cell._index)

#     # we simulate one more timestep (remember that the number of timesteps is
#     # num_times - 1)
#     # culture.simulate(num_times=2, culture_name="deactivated_culture_with_reactivated_cell_at_the_origin")
#     culture.reproduce(cell_index=first_cell._index, tic=6, dat_files=False)
#     # the fixture had simulation_steps = 6, so the last tic was 5 (the loop in
#     # simulate is: for i in range(1, num_times) ...)

#     # Cleanup code
#     db_files = ["data/deactivated_culture_with_reactivated_cell_at_the_origin.db", "deactivated_culture_with_reactivated_cell_at_the_origin.db"]
#     for db_file in db_files:
#         if os.path.exists(db_file):
#             os.remove(db_file)

#     # we assert that the first cell has not reproduced and it's been
#     # deactivated, i.e., it has no available space
#     assert first_cell.available_space is False
#     # and it's been removed from the active cells list
#     assert len(culture.active_cell_indexes) == 0


def test_deactivation_small():
    """We test that all cells are deactivated after trying to reproduce with
    no space available (but previously having space available, so that they
    have self.available_space = True)."""

    culture_name_temp = "deactivated_culture_fixture"

    try:
        # we create the culture
        deactivated_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=True,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        # we evolve it
        simulation_steps = 6
        deactivated_culture.simulate(
            num_times=simulation_steps, culture_name=culture_name_temp
        )

        # we deactivate its cells
        for cell in deactivated_culture.cells:
            cell.available_space = False

        deactivated_culture.active_cell_indexes = []

        # we assert that all cells were deactivated
        assert all(
            [
                cell.available_space is False
                for cell in deactivated_culture.cells
            ]
        )

        # we assert that the active cells list is empty
        assert len(deactivated_culture.active_cell_indexes) == 0

        # we identify the first cell
        first_cell = deactivated_culture.cells[0]

        # import ipdb; ipdb.set_trace()

        # we assert that it's located at the origin
        assert (
            first_cell.culture.cell_positions[first_cell._index].all()
            == np.array([0, 0, 0]).all()
        )

        # we reactivate the first cell (which at this time, t=6, does not have enough
        # space around it to reproduce)
        first_cell.available_space = True
        deactivated_culture.active_cell_indexes.append(first_cell._index)

        # we simulate one more timestep (remember that the number of timesteps is
        # num_times - 1)
        # deactivated_culture.simulate(num_times=2, culture_name="deactivated_culture_with_reactivated_cell_at_the_origin")
        deactivated_culture.reproduce(
            cell_index=first_cell._index, tic=6, dat_files=False
        )
        # the fixture had simulation_steps = 6, so the last tic was 5 (the loop in
        # simulate is: for i in range(1, num_times) ...)

        # we assert that the first cell has not reproduced and it's been
        # deactivated, i.e., it has no available space
        assert first_cell.available_space is False
        # and it's been removed from the active cells list
        assert len(deactivated_culture.active_cell_indexes) == 0

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)


@pytest.mark.slow
def test_deactivation_big():
    """We test the deactivation by checking that we have non-active cells in
    the culture when we simulate for a long time."""

    culture_name_temp = "deactivated_culture_fixture"

    try:
        # we create the culture
        deactivated_culture = Culture(
            adjacency_threshold=4,
            cell_radius=1,
            cell_max_repro_attempts=1000,
            first_cell_is_stem=True,
            prob_stem=0.7,
            prob_diff=0.0,
            rng_seed=110293658491283598,
            swap_probability=0.5,
        )
        # we evolve it
        simulation_steps = 15
        deactivated_culture.simulate(
            num_times=simulation_steps, culture_name=culture_name_temp
        )

        # assert that there are non-active cells
        # (and that we still have active cells, just in case)
        assert len(deactivated_culture.active_cell_indexes) < len(
            deactivated_culture.cells
        )
        assert len(deactivated_culture.active_cell_indexes) > 0
        # tipically, at t=15, we have something like
        # (total, active, stem, active_stem) = (1665.0, 830.0, 358.0, 146.0)
        # and the standard deviation of these numbers is around 10% of the mean
        assert len(deactivated_culture.active_cell_indexes) < 1000

    finally:
        # Cleanup code
        db_files = [f"data/{culture_name_temp}.db", f"{culture_name_temp}.db"]
        for db_file in db_files:
            if os.path.exists(db_file):
                os.remove(db_file)
