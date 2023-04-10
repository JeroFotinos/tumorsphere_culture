import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Enable LaTeX rendering in matplotlib
mpl.rcParams["text.usetex"] = True

# # por alguna razón, open() está en /home/nate/Devel/tumorsphere_culture
# # así que hay que darle el path desde ahí. Quizás porque el botón correr
# # de VS Code corre desde donde está ubicada la terminal. De todos modos,
# # podés chequearlo con lo siguiente:
# import os
# print(os.getcwd())

# Open the file and read the contents into a list
with open("./data/sim_1_time_measurement/dat_files/filename.txt", "r") as f:
    lines = f.readlines()

# Parse the time values from each line and store in a list
times = []
for line in lines:
    time_str = line.split(":")[1].strip()  # Extract the time string
    time_val = (
        float(time_str[:-7]) / 60
    )  # Remove ' seconds' and convert to float
    times.append(time_val)

# Compute the time differences
time_dif = np.zeros(len(times) - 1)
for n in range(1, len(times)):
    time_dif[n - 1] = times[n] - times[n - 1]

# Create a plot with the time values on the y-axis and the step number on the x-axis
steps = range(1, len(times))  # Generate a list of step numbers (1-indexed)
plt.plot(steps, time_dif, marker=".")

# we set the grid
plt.grid(color="green", linestyle="--", linewidth=0.5)

# Set y-axis scale to logarithmic
plt.yscale("log")

# Add labels and title
xlabel = r"$n$"
ylabel = r"$t(n) -t(n-1)$ [minutes]"
title = r"Time increment in calculating the next step (logarithmic scale)"
# Here, the r before the title string indicates that it is a raw string,
# which allows us to use backslashes to specify LaTeX commands.
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.title(title)

# Save the plot as a PNG file
plt.savefig(
    "./data/sim_1_time_measurement/post_processing/time_differences_by_step_log.png",
    dpi=600,
)
# the dpi (dots per inch) is set to 100 by default, but it's too low for me
