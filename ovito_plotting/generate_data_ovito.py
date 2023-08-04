"""Simulate an example culture and generate data in the appropriate format for Ovito.
"""

from tumorsphere.cells import *
from tumorsphere.culture import *

# culture.cell_positions[cell._position_index] replace for culture.cell_positions[cell._position_index]

def make_data_file(path, culture, i):
    path_to_write = f"{path}/ovito_data.{i}"
    with open(path_to_write, "w") as file_to_write:
        file_to_write.write(str(len(culture.cells)) + "\n")
        file_to_write.write(
            ' Lattice="1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0"Properties=species:S:1:pos:R:3:Color:r:1'
            + "\n"
        )

        for cell in culture.cells:  # csc activas
            if cell.is_stem and cell.available_space:
                line = (
                    "active_stem "
                    + str(culture.cell_positions[cell._position_index][0])
                    + " "
                    + str(culture.cell_positions[cell._position_index][1])
                    + " "
                    + str(culture.cell_positions[cell._position_index][2])
                    + " "
                    + "1"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # csc quiesc
            if cell.is_stem and (not cell.available_space):
                line = (
                    "quiesc_stem "
                    + str(culture.cell_positions[cell._position_index][0])
                    + " "
                    + str(culture.cell_positions[cell._position_index][1])
                    + " "
                    + str(culture.cell_positions[cell._position_index][2])
                    + " "
                    + "2"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # dcc activas
            if (not cell.is_stem) and cell.available_space:
                line = (
                    "active_diff "
                    + str(culture.cell_positions[cell._position_index][0])
                    + " "
                    + str(culture.cell_positions[cell._position_index][1])
                    + " "
                    + str(culture.cell_positions[cell._position_index][2])
                    + " "
                    + "3"
                    + "\n"
                )
                file_to_write.write(line)

        for cell in culture.cells:  # dcc quiesc
            if not (cell.is_stem or cell.available_space):
                line = (
                    "quiesc_diff "
                    + str(culture.cell_positions[cell._position_index][0])
                    + " "
                    + str(culture.cell_positions[cell._position_index][1])
                    + " "
                    + str(culture.cell_positions[cell._position_index][2])
                    + " "
                    + "4"
                    + "\n"
                )
                file_to_write.write(line)


# num_steps = 14

# csc_culture = Culture(
#     first_cell_is_stem=True,
#     prob_stem=0.7,
#     cell_max_repro_attempts=1000,
#     rng_seed=132450689123546892384,
# )

# make_data_file(csc_culture, 0)

# for i in range(num_steps + 1):
#     csc_culture.simulate(1)
#     make_data_file(csc_culture, i + 1)

folder = './ovito_plotting/huevo_o_no'


num_steps = 24

prob_diff = 0.0
prob_stem = 0.7
j = 1

culture_name = f"culture_pd={prob_diff}_ps={prob_stem}_realization_{j}.dat"

big_culture = CultureLite(
    first_cell_is_stem=True,
    prob_stem=0.7,
)

# this version just saves the final state due to changes in the culture and cell classes (Lite versions)
big_culture.simulate_with_persistent_data(
    num_times=num_steps,
    culture_name=culture_name,
    )
make_data_file(path=folder, culture=big_culture, i=num_steps)
