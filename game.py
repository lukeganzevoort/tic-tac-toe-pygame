import random
from typing import Optional

import numpy as np


class TicTacToe:
    def __init__(self) -> None:
        self.board: np.ndarray = np.zeros((3, 3), dtype=int)
        self.current_player: int = random.choice([1, 2])

    def move(self, row: int, col: int, player: int) -> bool:
        assert (
            player == self.current_player
        ), f"You played out of turn! {player} {self.current_player}"
        if self.board[row, col] == 0 and self.winner() is None:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player
            if self.winner() is not None:
                self.current_player = 0
            return True
        else:
            return False

    def winner(self) -> Optional[int]:
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
        if self.is_full():
            return 0
        return None

    def get_board(self) -> np.ndarray:
        return self.board

    # TODO: Make sure this works properly in a draw
    def is_full(self) -> bool:
        return not (self.board == 0).any()
