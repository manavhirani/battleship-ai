from board import board as board
from flags import *
import numpy as np
import random
from mcts import *

class player:
    def __init__(self, ID=TREE_BASED_AI, size=DEFAULT_SIZE) -> None:
        self.id = ID
        self.size = size
        self.board = board(size)
        self.score = 0
        self.ships = []
        self.ship_lens = []
        self.node = Node(self.board.hits)

    def __str__(self) -> str:
        # TODO
        lines = []
        line1 = ''
        return str(self.board)

    def fire_shot(self, shot):
        x,y = shot
        if self.board.grid[x][y] == 1:
            self.board.grid[x][y] = 2
            return True
        else:
            return False
    
    def valid_actions(self):
        unhit = np.where(self.board.hits == 0)
        return list(zip(unhit[0], unhit[1]))

    def get_shot(self) -> tuple:
        l = len(str(self.size**2-1))
        va = self.valid_actions()
        shot = random.choice(va)
        if self.id == HUMAN:
            print('CHOOSE YOUR SHOT FROM THE GRID:')
            lines = []
            c = 0
            pos = {}
            for i in range(self.size):
                line = []
                for j in range(self.size):
                    if (i,j) in va:
                        num = str(i) + str(j)
                        line.append(num.rjust(l))
                        pos[str(num)] = (i,j)
                    else:
                        line.append(' '*l)
                lines.append(' '.join(line))
            print('\n'.join(lines))
            while True:
                num = input('ENTER POSITION: ')
                try:
                    return pos[num]
                except:
                    print('ENTER VALID POSITION')

        if self.id == BASELINE_AI:
            shot = random.choice(va)

        if self.id == TREE_BASED_AI:
            # TODO: replace with MCTS_shot
            shot = va[np.argmax(rollout(self.node, self.ship_lens))]

        return shot
    
    def add_random_ships(self, ships):
        for ship in ships:
            size, orientation = ship
            x,y = self.board.get_ship_pos(size, bool(orientation))
            self.board.add_ship((x,y), size, bool(orientation))
            