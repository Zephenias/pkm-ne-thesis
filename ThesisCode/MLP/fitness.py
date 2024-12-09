#defining fitness function by what it checks.
#1. Pokemon levels
#2. Map exploration

#TODO: Read party levels and scale reward
#TODO: local map exploration reward = % of local map, finding new Area = +1 

#read memory
#pkm lvl 1 @ 0xD18C, 2 @ 0xD1B9, 3 @ 0xD1E4, 4 @ 0xD210, 5 @ 0xD23C, 6 @ 0xD268

from global_map import local_to_global, GLOBAL_MAP_SHAPE

def calculate_fitness(self):
    return