import os

# import filecmp
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import distance_matrix

from tumorsphere.core.simulation import Simulation


def test_no_overlap():
    """Test that no overlap occurs between the two cells of a culture."""
    cwd = Path(__file__).parent.resolve() / "data/"
    output_dir = cwd / "testing_simulate_no_overlap"

    # Ensure the output directory is clean before running the test
    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        # We create the directory (usually done by the CLI)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # We create the Simulation object
        sim = Simulation(
            first_cell_is_stem=True,
            prob_stem=[0.7],
            prob_diff=[0.0],
            num_of_realizations=1,
            num_of_steps_per_realization=7,
        )

        sim.simulate_single_culture(
            sql=True,
            output_dir=output_dir,
        )

        # sim.simulate_parallel(
        #     sql=True,
        #     number_of_processes=1,
        #     output_dir=output_dir,
        # )

        # Check if the output directory was created
        assert output_dir.exists(), "The output directory was not created."

        assert sim.cultures is not None, "No cultures were created."
        assert sim.cultures != {}, "No cultures were created."
        assert len(sim.cultures) == 1, "More than one culture was created."

        # We convert dict_values to a list and access the first item
        culture = list(sim.cultures.values())[0]

        # We use culture.cell_positions to compute the matrix of distances
        # between cells
        cell_positions = culture.cell_positions
        dist_matrix = distance_matrix(cell_positions, cell_positions)

        # We set diagonal elements to a high enough value to ignore
        # self-comparisons
        np.fill_diagonal(dist_matrix, 2 * culture.cell_radius + 10)

        # Now we check that every element of the distance matrix is greater
        # than 2 * culture.cell_radius, or that it's within the numerical
        # precision of the machine
        dist_aprox_two_radii = np.isclose(dist_matrix, 2 * culture.cell_radius)
        dist_greater_than_two_radii = dist_matrix > 2 * culture.cell_radius

        assert (
            np.logical_or(dist_aprox_two_radii, dist_greater_than_two_radii)
        ).all(), "Overlap between cells was detected."
        # Note: I didn't use the pipe operator | here for compatibility across
        # Python versions.

    finally:
        # Cleanup: remove the output directory after the test
        shutil.rmtree(output_dir)
