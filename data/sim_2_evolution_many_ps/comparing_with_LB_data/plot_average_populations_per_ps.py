import os

import matplotlib.pyplot as plt
import numpy as np

# Set the path to the directory containing the data files
# data of my simulations
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages/"
# data of Lucas simulations
dir_lucas = "/home/nate/Devel/tumorsphere_culture/data/data_Lucas"

# Set the value of p for which to plot the data
p = 0.7

# Find my data file for the specified value of p
data_file = None
for file_name in os.listdir(data_dir):
    if file_name.startswith("average-ps-{}".format(p)):
        data_file = os.path.join(data_dir, file_name)
        break

# Find LB's data file for the specified value of p
lucas_file = None
for file in os.listdir(dir_lucas):
    if file.startswith("{}".format(p)):
        lucas_file = os.path.join(dir_lucas, file)
        break

# Read the data from the files
data_jero = np.loadtxt(data_file, delimiter=",", skiprows=0)
# skiprows = 1 es para el caso en el que las columnas del csv tienen título
data_lucas = np.loadtxt(lucas_file, skiprows=0) #  delimiter="   ",


# Extract the columns of interest
time = data_jero[:, 0]
total_cells = data_jero[:, 1]
active_cells = data_jero[:, 2]
total_stem_cells = data_jero[:, 3]
active_stem_cells = data_jero[:, 4]

time_lucas = data_lucas[:, 0]
col2_lucas = data_lucas[:, 1]
col3_lucas = data_lucas[:, 2]

# Plot the curves
fig, ax = plt.subplots()

ax.plot(time_lucas, col2_lucas, 'o', label="Lucas Col 2")
ax.plot(time_lucas, col3_lucas, 'o', label="Lucas Col 3")

ax.plot(time, total_cells, marker=".", label="Total Cells")
ax.plot(time, active_cells, marker=".", label="Active Cells")
ax.plot(time, total_stem_cells, marker=".", label="Total Stem Cells")
ax.plot(time, active_stem_cells, marker=".", label="Active Stem Cells")
ax.set_xlabel("Time")
ax.set_ylabel("Cell Count")
ax.set_title("Evolution of Tumor Cell Populations")
ax.legend()

plt.yscale("log")


# we set the grid
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# to see the figure
# plt.show()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/comparing_with_LB_data/comparison-ps-{}.png".format(
        p
    ),
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
