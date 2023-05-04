"""Active cells are evaluated as active stem cells in the case p_s = 1,
just for "historical" reasons.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Set the path to the directory containing the data files
data_dir = "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/averages/"

# Set the value of p for which to plot the data
p = 1

# Find the data file for the specified value of p
data_file = None
for file_name in os.listdir(data_dir):
    if file_name.startswith("average-ps-{}".format(p)):
        data_file = os.path.join(data_dir, file_name)
        break

# Read the data from the file
data = np.loadtxt(data_file, delimiter=",", skiprows=0)
# skiprows = 1 es para el caso en el que las columnas del csv tienen título

# Extract the columns of interest
time = data[:, 0]
# total_cells = data[:, 1]
# active_cells = data[:, 2]
# total_stem_cells = data[:, 3]
active_stem_cells = data[:, 4]

time = np.log(time)
active_stem_cells = np.log(active_stem_cells)

# Plot the curves
fig, ax = plt.subplots()
# ax.plot(time, total_cells, marker=".", label="Total Cells")
# ax.plot(time, active_cells, marker=".", label="Active Cells")
# ax.plot(time, total_stem_cells, marker=".", label="Total Stem Cells")
ax.plot(time, active_stem_cells, marker=".", label="Active Stem Cells")

# Set y-axis scale to logarithmic
# plt.yscale("log")

# Perform linear regression on the 25-th step onwards
last_steps = time[-10:]
last_times = active_stem_cells[-10:]
slope, intercept, r_value, p_value, std_err = linregress(
    last_steps, last_times
)

# Add the linear fit to the plot with label and legend
fit_label = (
    f"Linear fit: $\log[n_s(t)] = {slope:.2f} ~ \log(t) + {intercept:.2f}$"
)
plt.plot(last_steps, slope * last_steps + intercept, label=fit_label)
plt.legend()

# Add text to the plot to display the fit statistics
text_xpos = time[-1] - 1  # Position the text near the end of the data
text_ypos = (
    active_stem_cells[-1] * 0.1
)  # Position the text 10% up from the lowest point
text = f"$r = {r_value:.4f}$\n$p = {p_value:.4f}$\n$\sigma = {std_err:.4f}$"
plt.text(text_xpos, text_ypos, text)

ax.set_xlabel("$\log(t)$")
ax.set_ylabel("$\log[n_s(t)]$")
ax.set_title("Evolution of Active cell number $n_s$ in time")
ax.legend()

# we set the grid
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# to see the figure
# plt.show()

# Save the plot as a PNG file
plt.savefig(
    "/home/nate/Devel/tumorsphere_culture/data/sim_2_evolution_many_ps/scaling/scaling_plots/average-active-log.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
