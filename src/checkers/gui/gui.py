import tkinter
from tkinter import *

from src.checkers.game import Game

TILE_BORDER = 0.75
CHECKER_BORDER = 4

COLUMNS_COUNT = 8
ROWS_COUNT = 8

TILE_HEIGHT = 31

TILE_WIDTH = 31


class Gui(Canvas):

	def __init__(self, game, red_name, grey_name):

		self.game = game
		self.root = tkinter.Tk()
		self.redScoreBoard = Label(self.root, text="Red: %s" % red_name)
		self.greyScoreBoard = Label(self.root, text="Grey: %s" % grey_name)
		Canvas.__init__(self, self.root, bg="grey", height=250, width=250)
		self.pack()
		self.redScoreBoard.pack()
		self.greyScoreBoard.pack()

		self.root.minsize(250, 350)

		self.render_tiles()
		self.render_checkers()
		self.root.mainloop()

	# Description: Function creates white and black tiles for the game board
	def render_tiles(self):
		for i in range(0, COLUMNS_COUNT):
			x1 = (i * TILE_WIDTH) + TILE_BORDER
			x2 = ((i + 1) * TILE_WIDTH) - TILE_BORDER
			for j in range(0, ROWS_COUNT):
				y1 = (j * TILE_HEIGHT) + TILE_BORDER
				y2 = ((j + 1) * TILE_HEIGHT) - TILE_BORDER

				if (i + j) % 2 == 0:
					idVal = self.create_rectangle(x1, y1, x2, y2, fill="white")
				else:
					idVal = self.create_rectangle(x1, y1, x2, y2, fill="black")
				# if idVal != 0:
				# 	self.board.append((idVal, j, i, x1, x2, y1, y2))

	def render_checkers(self):
		for piece in self.game.board.pieces:
			row, column = self.row_and_column(piece.position)

			x1, x2, y1, y2 = self.coordinates(row, column)

			if piece.player == 1:
				color = "white"
			else:
				color = "grey"

			self.create_oval(x1, y1, x2, y2, fill=color)
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


if __name__ == "__main__":
	game = Game()
	gui = Gui(game, 'player-1', 'player-7')