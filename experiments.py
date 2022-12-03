from battleground import battleground as bg
from flags import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_experiment(size):
    ai_wins = 0
    baseline_wins = 0

    for i in range(100):
        n = 0
        while True:
            n += 1
            try:
                ground = bg(size=size, player1=TREE_BASED_AI, player2=BASELINE_AI)
                ground.init_random_ships(np.random.randint(3,6))
                print()
                print(ground)
                ground.turn = not ground.turn
                print()
                print(ground)
                ground.turn = not ground.turn
                print('\rInitialized ships in', n, 'attempts', end='')
                n = 0
                break
            except Exception as e:
                if n == 30:
                    print('FAILED TO INITIALIZE SHIPS AFTER 30 ATTEMPTS')
                    break
                print(e)
        moves = 0

        while True:
            try:        
                ground.fire_shot()
                if ground.turn:
                    print(ground)
                ground.turn = not ground.turn
                if ground.is_game_over():
                    print('GAME OVER:', end=' ')
                    if ground.winner:
                        print('PLAYER 1 WINS')
                        ai_wins += 1
                    else:
                        print('PLAYER 2 WINS')
                        baseline_wins += 1
                    ground.turn = True
                    print(ground)
                    ground.turn = False
                    print(ground)
                    break
                # x = input('PRESS ENTER TO GO TO THE NEXT TURN')
                moves += 1
            except Exception as e:
                if moves == 40:
                    break
                print('ERROR')
    return (ai_wins, baseline_wins)

ai_wins = []
baseline_wins = []

for i in range(6,11):
    x,y = run_experiment(i)
    ai_wins.append(x)
    baseline_wins.append(y)

print(ai_wins)
print(baseline_wins)