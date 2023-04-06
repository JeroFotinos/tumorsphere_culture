"""This program plots the simulationdata using the pickled dictionary
of average data. The paths are thought so the script is used in the
cluster
"""
import pickle

# open the pickled dictionary file
with open("tumorsphere/sim_average_data.pkl", "rb") as f:
    my_dict = pickle.load(f)

import matplotlib.pyplot as plt

# create a figure and axis objects
fig, ax = plt.subplots()

# plot each row of the array with custom labels and colors
ax.plot(my_dict[0, :, 0], label="Total", color="blue")
ax.plot(
    my_dict[1, :, 0],
    label="Total active",
    color="green",
)
ax.plot(my_dict[2, :, 0], label="Stem", color="orange")
ax.plot(my_dict[3, :, 0], label="Active stem", color="red")

# set the title and axis labels
ax.set_title("Average evolution of culture")
ax.set_xlabel("Time step")
ax.set_ylabel("Number of cells")

# create a legend and display the plot
ax.legend()

plt.show()
