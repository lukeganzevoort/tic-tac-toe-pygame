import random
from typing import Optional

import numpy as np


class TicTacToe:
    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((3, 3), dtype=int)
        self.current_player: int = random.choice([1, 2])

    def move(self, row: int, col: int) -> bool:
        if self.board[row, col] == 0 and self.winner() is None:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player
            return True
        else:
            return False

    def winner(self) -> Optional[int]:
        if self.is_full():
            return 0
        for player in [1, 2]:
            for i in range(3):
                if np.all(self.board[i, :] == player) or np.all(
                    self.board[:, i] == player
                ):
                    return player
            if np.all(np.diag(self.board) == player) or np.all(
                np.diag(np.fliplr(self.board)) == player
            ):
                return player
        return None

    def get_board(self) -> np.ndarray:
        return self.board

    # TODO: Make sure this works properly in a draw
    def is_full(self) -> bool:
        return not (self.board == 0).any()
