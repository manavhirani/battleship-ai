from text_screens import screens as sc
import sys, os, time
from battleground import *

def start_game():
    screen = sc()
    option = screen.print_home_page()
    environment = screen.text_based_interface(option)
    play_game(environment)
    
def play_game(environment):
    screen = sc()
    role1, role2, size = environment
    ground = battleground(role1, role2, size)
    ground.init_random_ships(np.random.randint(3,6))
    print('Initialized game board succesfully, press ENTER to start!')
    while True:
        # screen.clear_screen()
        if ground.is_game_over():
            print('GAME OVER: ', end='')
            if ground.turn: 
                print('PLAYER 2 WINS')
            else: 
                print('PLAYER 1 WINS')
            ground.turn = True
            print(ground)
            ground.turn = not ground.turn
            print(ground)
            print(f'FINAL SCORES:\nPLAYER 1: {ground.p1.score}\nPLAYER 2: {ground.p2.score}')
            break

        print(ground)
        ground.fire_shot()
        x = input('PRESS ENTER TO CONTINUE')
        ground.turn = not ground.turn
    
if __name__ == '__main__':
    start_game()