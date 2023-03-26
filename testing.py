from core import *

test_culture = Culture()
# print(test_culture.cells[0].position)
test_culture.simulate(1)
print(len(test_culture.cells))
# print(test_culture.cells)
test_culture.plot_culture_spheres()
