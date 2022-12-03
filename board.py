import numpy as np
import random
from flags import *


class board:
    def __init__(self, size=DEFAULT_SIZE) -> None:
        self.size = size
        self.grid = np.zeros((size,size), int)
        self.hits = np.zeros((size,size), int)

    def __str__(self) -> str:
        grid = self.grid
        hits = self.hits
        lines = []
        lines.append('BOARD'.ljust(self.size*2)+'   '+'ATTACKS'.ljust(self.size*2))
        for row in range(self.size):
            icon1 = ['*', '$', 'X']
            icon2 = ['.', 'O', 'X']
            icon3 = ['.','X','_']
            line = ' '.join(icon2[col] for col in grid[row]) + '    ' + ' '.join(icon3[col] for col in hits[row])
            lines.append(line)
        return '\n'.join(lines)
    
    def get_ship_pos(self, size, orientation):
        if orientation:
            positions = []
            for i in range(self.size):
                for j in range(self.size-size):
                    locs = [x for x in self.grid[i][j:j+size]]
                    if 1 not in locs:
                        positions.append((i,j))
            return random.choice(positions)
        else:
            positions = []
            for i in range(self.size-size):
                for j in range(self.size):
                    locs = [x[j] for x in self.grid[i:i+size]]
                    if 1 not in locs:
                        positions.append((i,j))
            return random.choice(positions)
    
    def add_ship(self, position, size, orientation):
        if orientation:
            x,y = position
            for i in range(y, y+size):
                self.grid[x][i] = 1
        else:
            x,y = position
            for i in range(x,x+size):
                self.grid[i][y] = 1

class ship:
    def __init__(self, size, orientation, position) -> None:
        self.shape = np.ones((size, 1))
        self.orientation = orientation