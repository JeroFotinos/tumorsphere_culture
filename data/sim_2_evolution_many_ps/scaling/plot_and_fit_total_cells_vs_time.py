import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Set the path to the directory containing the data files
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages/"

# Set of values for p for which are available to plot
# p = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
p = [1.0]

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
    time = np.log(time)  # log scale in the x axis
    total_cells = np.log(total_cells)  # log scale in the y axis

# we set the grid
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Plot the curves
fig, ax = plt.subplots()

for p_index in range(len(p)):
    ax.plot(
        time[p_index],
        total_cells[p_index],
        marker=".",
        label=f"$p_s = {p[p_index]}$",
        color=plt.cm.viridis(p_index / len(p)),
    )  # , label="Total Cells"
    # ax.plot(time[p_index], active_cells[p_index], label="Active Cells")
    # ax.plot(time[p_index], total_stem_cells[p_index], label=f'$p_s = {p[p_index]}$') # , label="Total Stem Cells"
    # ax.plot(time[p_index], active_stem_cells[p_index], label=f'$p_s = {p[p_index]}$') # , label="Active Stem Cells"

    # Perform linear regression on the 10-th step onwards
    last_steps = time[p_index][10:]
    last_times = total_cells[p_index][10:]
    slope, intercept, r_value, p_value, std_err = linregress(
        last_steps, last_times
    )

    # Add the linear fit to the plot with label and legend
    fit_label = (
        f"Linear fit: $\log[n(t)] = {slope:.2f} ~ \log(t) {intercept:.2f}$"
    )
    plt.plot(last_steps, slope * last_steps + intercept, label=fit_label)
    plt.legend()

    # Add text to the plot to display the fit statistics
    text_xpos = (
        time[p_index][-1] - 1
    )  # Position the text near the end of the data
    text_ypos = (
        total_cells[p_index][-1] * 0.1
    )  # Position the text 10% up from the lowest point
    text = (
        f"$r$ = {r_value:.4f}\n$p$ = {p_value:.4f}\n$\sigma$ = {std_err:.4f}"
    )
    plt.text(text_xpos, text_ypos, text)


# we set the grid
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Set x-axis log
# plt.xscale("log")

# Set y-axis scale to logarithmic
# plt.yscale("log")

ax.set_xlabel("$\log(t)$")
ax.set_ylabel("$\log[n(t)]$")
ax.set_title("Evolution of Total Cancer Cells in time")
ax.legend()

# to see the figure
# plt.show()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/scaling/scaling_plots/average-total-cells-log.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
