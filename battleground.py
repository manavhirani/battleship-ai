from player import *
from flags import *
import numpy as np

class battleground:
    
    def __init__(self, player1=BASELINE_AI, player2=TREE_BASED_AI, size=DEFAULT_SIZE, num_ships=DEFAULT_SHIPS):
        self.size = size
        self.p1 = player(player1, size)
        self.p2 = player(player2, size)
        self.num_ships = num_ships
        self.turn = True
        self.winner = None
        self.ship_sizes = None

    def play_turn(self, action) -> bool:
        if self.turn:
            self.p1.fire_shot(action)
        else:
            self.p1.fire_shot(action)
        self.turn = not self.turn
    
    def init_random_ships(self, num_ships):
        self.ship_sizes = np.random.randint(self.size//3, 2*self.size//3+1, size=num_ships)
        self.p1.ship_lens = self.p2.ship_lens = self.ship_sizes
        self.p1.ships = list(zip(self.ship_sizes, np.random.randint(2, size=num_ships)))
        self.p2.ships = list(zip(self.ship_sizes, np.random.randint(2, size=num_ships)))
        self.p1.add_random_ships(self.p1.ships)
        self.p2.add_random_ships(self.p2.ships)
    
    def is_game_over(self) -> bool:
        if self.turn:
            for row in self.p1.board.grid:
                for col in row:
                    if col == 1:
                        return False
        else:
            for row in self.p2.board.grid:
                for col in row:
                    if col == 1:
                        return False

        self.winner = not self.turn
        return True

    def fire_shot(self):
        if self.turn:
            x,y = self.p1.get_shot()
            success = self.p2.fire_shot((x,y))
            print(f'Firing shot {x},{y}')
            if success:
                print('HIT')
                self.p1.board.hits[x][y] = 1
                self.p1.score += 1
            else:
                print('MISS')
                self.p1.board.hits[x][y] = 2
        else:
            x,y = self.p2.get_shot()
            success = self.p1.fire_shot((x,y))
            print(f'Firing shot {x},{y}')
            if success:
                print('HIT')
                self.p2.board.hits[x][y] = 1
                self.p2.score += 1
            else:
                print('MISS')
                self.p2.board.hits[x][y] = 2
        return success

    def __str__(self) -> str:
        player_types = ['HUMAN', 'BASELINE_AI', 'TREE_BASED_AI']
        lines = []
        lines.append('-'*(self.size*4+2))
        if self.turn:
            lines.append(f'{player_types[self.p1.id-1]} PLAYER 1:')
            lines.append(str(self.p1))
            return '\n'.join(lines)
        else:
            lines.append(f'{player_types[self.p2.id-1]} PLAYER 2:')
            lines.append(str(self.p2))
            return '\n'.join(lines)