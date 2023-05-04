import os

import matplotlib.pyplot as plt
import numpy as np

# set matplotlib style
plt.style.use("ggplot")
plt.rcParams["axes.edgecolor"] = "darkgray"
plt.rcParams["axes.linewidth"] = 0.8

# Set the path to the directory containing the data files
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages/"

# Set of values for p for which are available to plot
p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

# p with reversed order
p = p[::-1]

# Find the data files for the specified values of p
data_files = []
for p_index in range(len(p)):
    files_for_p = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith("average-ps-{}".format(p[p_index])):
            files_for_p.append(file_name)
    files_for_p.sort()  # sort the file names for this p value
    for file_name in files_for_p:
        data_files.append(os.path.join(data_dir, file_name))

# print(data_files)

# Read the data from the files
data = []
for file_index in range(len(p)):
    data.append(np.loadtxt(data_files[file_index], delimiter=",", skiprows=0))
# skiprows = 1 es para el caso en el que las columnas del csv tienen título

# Extract the columns of interest
time = []
total_cells = []
active_cells = []
total_stem_cells = []
active_stem_cells = []

for p_index in range(len(p)):
    time.append(data[p_index][:, 0])
    total_cells.append(data[p_index][:, 1])
    active_cells.append(data[p_index][:, 2])
    total_stem_cells.append(data[p_index][:, 3])
    active_stem_cells.append(data[p_index][:, 4])

# Plot the curves
fig, ax = plt.subplots()

for p_index in range(len(p)):
    # ax.plot(time[p_index], total_cells[p_index], label="Total Cells")
    # ax.plot(time[p_index], active_cells[p_index], label="Active Cells")
    ax.plot(
        time[p_index],
        total_stem_cells[p_index],
        marker=".",
        label=f"$p_s = {p[p_index]}$",
        color=plt.cm.magma(p_index / len(p)),
    )  # , label="Total Stem Cells"
    # ax.plot(time[p_index], active_stem_cells[p_index], label="Active Stem Cells")

# we set the grid
# plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Set y-axis scale to logarithmic
plt.yscale("log")

ax.set_xlabel("Time")
ax.set_ylabel("Total Stem Cells")
ax.set_title("Evolution of Total Cancer Stem Cells")
ax.legend(prop={"size": 8.5})

# to see the figure
# plt.show()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages_plots/average-total-stem-log.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
