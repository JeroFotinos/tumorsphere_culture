from cells import *
from culture import *

csc_culture = Culture(first_cell_type='csc')
csc_culture.simulate(10)
print(len(csc_culture.cells))
csc_culture.plot_culture_spheres()