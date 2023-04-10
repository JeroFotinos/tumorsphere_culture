import os

import matplotlib.pyplot as plt
import numpy as np

# Set the path to the directory containing the data files
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/p_infty_vs_ps_averages/"

# Set of values for p for which are available to plot
list_of_steps = np.arange(25, 31, 1)
# list_of_steps = [30]

# Find the data files for the specified values of p
data_files = []
for step_index in range(len(list_of_steps)):
    files_for_i = []
    for file_name in os.listdir(data_dir):
        if file_name.startswith("average-p_infty_vs_ps-t-{}".format(list_of_steps[step_index])):
            files_for_i.append(file_name)
    files_for_i.sort()  # sort the file names for this p value
    for file_name in files_for_i:
        data_files.append(os.path.join(data_dir, file_name))

# print(data_files)

# Read the data from the files
data = []
for file_index in range(len(list_of_steps)):
    data.append(np.loadtxt(data_files[file_index], delimiter=",", skiprows=0))
# skiprows = 1 es para el caso en el que las columnas del csv tienen título

# Extract the columns of interest
ps = []
p_infty = []

# we order the arrays for growing ps
for step_index in range(len(list_of_steps)):
    indices_to_sort_array = np.array(np.argsort(data[step_index][:, 0]))
    ps.append(data[step_index][indices_to_sort_array, 0])
    p_infty.append(data[step_index][indices_to_sort_array, 1])

# Plot the curves
fig, ax = plt.subplots()

for step_index in range(len(list_of_steps)):
    ax.plot(
        ps[step_index],
        p_infty[step_index],
        marker=".",
        label=f"$t = {list_of_steps[step_index]}$",
        color=plt.cm.viridis(step_index / len(list_of_steps)),
    )

# we set the grid
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Set y-axis scale to logarithmic
# plt.yscale("log")

ax.set_xlabel("$p_s$")
ax.set_ylabel("$P_\infty (p_s)$")
ax.set_title("Probability of having an active CSC vs Probability of self-replication")
ax.legend()

# to see the figure
# plt.show()

# This make the x and y labels smaller and closer to the axes
# plt.tight_layout()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages_plots/average-p_infty_vs_ps.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
