"""Simulate an example culture and generate data in the appropriate format for Ovito.
"""

from tumorsphere.cells import *
from tumorsphere.culture import *


def make_data_file(culture, i):
    with open(f"ovito_data.{i}", "w") as file_to_write:
        file_to_write.write(str(len(culture.cells)) + "\n")
        file_to_write.write(
            ' Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"Properties=species:S:1:pos:R:3:Color:r:1'
            + "\n"
        )

        for cell in culture.cells:  # csc activas
            if cell.is_stem and cell.available_space:
                line = (
                    "active_stem "
                    + str(cell.position[0])
                    + " "
                    + str(cell.position[1])
                    + " "
                    + str(cell.position[2])
                    + " "
                    + "1"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # csc quiesc
            if cell.is_stem and (not cell.available_space):
                line = (
                    "quiesc_stem "
                    + str(cell.position[0])
                    + " "
                    + str(cell.position[1])
                    + " "
                    + str(cell.position[2])
                    + " "
                    + "2"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # dcc activas
            if (not cell.is_stem) and cell.available_space:
                line = (
                    "active_diff "
                    + str(cell.position[0])
                    + " "
                    + str(cell.position[1])
                    + " "
                    + str(cell.position[2])
                    + " "
                    + "3"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # dcc quiesc
            if not (cell.is_stem or cell.available_space):
                line = (
                    "quiesc_diff "
                    + str(cell.position[0])
                    + " "
                    + str(cell.position[1])
                    + " "
                    + str(cell.position[2])
                    + " "
                    + "4"
                    + "\n"
                )
                file_to_write.write(line)


num_steps = 14

csc_culture = Culture(
    first_cell_is_stem=True,
    prob_stem=0.7,
    cell_max_repro_attempts=1000,
    rng_seed=132450689123546892384,
)

make_data_file(csc_culture, 0)

for i in range(num_steps + 1):
    csc_culture.simulate(1)
    make_data_file(csc_culture, i + 1)
