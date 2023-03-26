from cells import *
from culture import *

test_culture1 = Culture(cell_max_repro_attempts=50)
# print(test_culture.cells[0].position)
test_culture1.simulate(10)
print(len(test_culture1.cells))
# print(test_culture.cells)
test_culture1.plot_culture_spheres()
