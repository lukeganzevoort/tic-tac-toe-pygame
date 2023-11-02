import random

import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = random.choice([1, 2])

    def move(self, row, col):
        if self.board[row, col] == 0 and self.winner() is None:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        else:
            return False

    def winner(self):
        for i in range(3):
            if np.all(self.board[i, :] == self.current_player) or np.all(
                self.board[:, i] == self.current_player
            ):
                return self.current_player
        if np.all(np.diag(self.board) == self.current_player) or np.all(
            np.diag(np.fliplr(self.board)) == self.current_player
        ):
            return self.current_player
        return None

    def get_board(self):
        return self.board

    def is_full(self):
        return not (self.board == 0).any()
