"""Script to generate data for animating with Ovito.

This script loads a culture object from a pickle file and writes to a .dat
file. First, it writes the total number of cells. Then, in another line, it
writes some quite obscure configuration line. Finally, for each cell in the
culture, it writes the type of cell, its position, and an integer for specifing
the color when plotting. There is a bijection between color and type of a cell.

DEPRECATED: use '../ovito_plotting/generate_data_ovito.py'.
"""

from tumorsphere.cells import *
from tumorsphere.culture import *

import pickle
import glob

num_steps = 10
for i in range(num_steps+1):    
    with open(f'./csc_culture_{i}.pkl', 'rb') as pickle_file, open(f"ovito_data.{i}","w") as file_to_write:
        culture_object = pickle.load(pickle_file)
        print("Number of cells:", len(culture_object.cells))
        file_to_write.write(str(len(culture_object.cells)) + "\n" )
        file_to_write.write(" Lattice=\"1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\"Properties=species:S:1:pos:R:3:Color:r:1"+"\n")
        
        for cell in culture_object.cells: # csc activas
            if (cell.is_stem and cell.available_space):
                line = "active_stem " + str(cell.position[0]) + " " + str(cell.position[1]) + " " + str(cell.position[2]) + " " + "1" + "\n"
                file_to_write.write(line)
                print(line)
        for cell in culture_object.cells: # csc quiesc
            if (cell.is_stem and (not cell.available_space)):
                line = "quiesc_stem " + str(cell.position[0]) + " " + str(cell.position[1]) + " " + str(cell.position[2]) + " " + "2" + "\n"
                file_to_write.write(line)
                print(line)
        for cell in culture_object.cells: # dcc activas
            if ((not cell.is_stem) and cell.available_space):
                line = "active_diff " + str(cell.position[0]) + " " + str(cell.position[1]) + " " + str(cell.position[2]) + " " + "3" + "\n"
                file_to_write.write(line)
                print(line)
        for cell in culture_object.cells: # dcc quiesc
            if not (cell.is_stem or cell.available_space):
                line = "quiesc_diff " + str(cell.position[0]) + " " + str(cell.position[1]) + " " + str(cell.position[2]) + " " + "4" + "\n"
                file_to_write.write(line)
                print(line)