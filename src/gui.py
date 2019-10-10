import tkinter
import random
from tkinter import *

from src.agent import Agent
from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.game import Game
from src.checkers.state_encoder import StateEncoder
from src.model import Residual_CNN
from src.my_configs import config

TILE_BORDER = 0.75
CHECKER_BORDER = 4

COLUMNS_COUNT = 8
ROWS_COUNT = 8

TILE_HEIGHT = 31

TILE_WIDTH = 31


class Gui(Canvas):

	def __init__(self, game, player1, player2):
		player1_starts = random.choice([True, False])
		if player1_starts:
			self.current_player = player1
		else:
			self.current_player = player2

		self.player2 = player2
		self.player1 = player1
		self.game = game
		self.root = tkinter.Tk()
		if player1_starts:
			player1_name = player1.name
			player2_name = player2.name
		else:
			player1_name = player2.name
			player2_name = player1.name
		self.redScoreBoard = Label(self.root, text="White: %s" % player1_name)
		self.greyScoreBoard = Label(self.root, text="Black: %s" % player2_name)
		Canvas.__init__(self, self.root, bg="grey", height=250, width=250)
		self.pack()
		self.redScoreBoard.pack()
		self.greyScoreBoard.pack()

		self.root.minsize(250, 350)
		self.checkers = []
		self.render_tiles()
		self.render_checkers()
		self.root.after(2000, self.do_move)
		self.root.mainloop()

	def do_move(self):
		action, pi, value = self.current_player.act(self.game, 0)
		self.game = self.game.move_with_additional_jumps(action)
		self.cleanup()
		self.render_checkers()
		if self.game.is_over():
			print("Game is over, winner is: " + str(self.game.get_winner()))
		else:
			if self.current_player == self.player1:
				self.current_player = self.player2
			else:
				self.current_player = self.player1
			self.root.after(1000, self.do_move)

	def cleanup(self):
		for id in self.checkers:
			self.delete(id)

	# Description: Function creates white and black tiles for the game board
	def render_tiles(self):
		for i in range(0, COLUMNS_COUNT):
			x1 = (i * TILE_WIDTH) + TILE_BORDER
			x2 = ((i + 1) * TILE_WIDTH) - TILE_BORDER
			for j in range(0, ROWS_COUNT):
				y1 = (j * TILE_HEIGHT) + TILE_BORDER
				y2 = ((j + 1) * TILE_HEIGHT) - TILE_BORDER

				if (i + j) % 2 == 0:
					self.create_rectangle(x1, y1, x2, y2, fill="white")
				else:
					self.create_rectangle(x1, y1, x2, y2, fill="black")

	def render_checkers(self):
		for piece in self.game.board.pieces:
			if piece.captured:
				continue
			row, column = self.row_and_column(piece.position)

			x1, x2, y1, y2 = self.coordinates(row, column)

			if piece.player == 1:
				if piece.king:
					color = "yellow"
				else:
					color = "white"
			else:
				if piece.king:
					color = "brown"
				else:
					color = "grey"

			oval = self.create_oval(x1, y1, x2, y2, fill=color)
			self.checkers.append(oval)
			# self.tag_bind(idTag, "<ButtonPress-1>", self.processCheckerClick)

	def coordinates(self, row, column):
		y1 = ((7 - row) * TILE_WIDTH) + CHECKER_BORDER
		y2 = (((7 - row) + 1) * TILE_WIDTH) - CHECKER_BORDER
		x1 = (column * TILE_WIDTH) + CHECKER_BORDER
		x2 = ((column + 1) * TILE_WIDTH) - CHECKER_BORDER

		return x1, x2, y1, y2

	def row_and_column(self, position):
		row = (position - 1) // 4
		column = (3 - ((position - 1) % 4)) * 2
		if row % 2 == 1:
			column = column + 1
		return row, column


def read_agent(version):
	nn = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
							config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])
	m_tmp = nn.read(version)
	nn.model.set_weights(m_tmp.get_weights())
	player = Agent(nn, ActionEncoder(DirectionResolver()), StateEncoder(), name='player' + str(version),
						   config=config)
	return player


if __name__ == "__main__":
	game = Game()
	player1 = read_agent(config['GUI_PLAYERS'][0])
	player2 = read_agent(config['GUI_PLAYERS'][1])
	gui = Gui(game, player1, player2)

