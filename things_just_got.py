from cells import *
from culture import *

dcc_culture = Culture(first_cell_type='dcc')
dcc_culture.simulate(1)
print(len(dcc_culture.cells))
dcc_culture.plot_culture_spheres()