from flags import *
from ascii_graphics import *
import sys, os, time

class screens:

	def __init__(self) -> None:
		pass

	def print_home_page(self, line_delay=0) -> int:
		print('WELCOME TO BATTLESHIP')
		self.print_art(line_delay)
		self.print_main_menu()
		return self.get_option()
	
	def text_based_interface(self, option):
		if OPTIONS[option - 1] == 'START GAME':
			environment = self.get_environment()
		return environment

	def get_option(self):
		option = self.get_valid_input(range(1,1+len(OPTIONS)),'CHOOSE OPTION: ', int)
		return option

	def get_environment(self):
		self.print_game_menu()
		role1 = self.get_valid_input(range(1,1+len(ROLES)), 'PLAYER 1 ROLE: ', int)
		role2 = self.get_valid_input(range(1,1+len(ROLES)), 'PLAYER 2 ROLE: ', int)
		size = self.get_valid_input(range(5, 15), 'BOARD SIZE: ', int)
		return (role1, role2, size)

	def get_valid_input(self, constraints, prompt, dtype):
		while True:
			choice = input(prompt)
			try:
				choice = dtype(choice)
			except ValueError:
				print('Invalid choice! Try again!')
			if choice in constraints:
				return choice
			else:
				print('Invalid choice! Try again!')

	def set_environment(self):
		pass

	def print_main_menu(self):
		for i in range(len(OPTIONS)):
			print(f'[{i + 1}] {OPTIONS[i]}')

	def print_game_menu(self):
		for i in range(len(ROLES)):
			print(f'[{i + 1}] {ROLES[i]}')

	def print_art(self, t):
		split_art = (FONT_ASCII+'\n'+SHIP_ASCII).split('\n')
		for splits in split_art:
			print(f'\t{splits}')
			time.sleep(t)

	def clear_screen(self):
		if os.name == 'nt': os.system('cls')
		else: os.system('clear')