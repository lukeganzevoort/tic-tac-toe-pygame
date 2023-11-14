import json
import random
import time
from collections import deque
from typing import Any, Optional, Union

import numpy as np
import torch
from numpy._typing import NDArray

import api_client
from helper import plot
from model import Linear_QNet, QTrainer, nn

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

# State = list[int]


def get_params():
    with open("params.json", "r") as f:
        params = json.load(f)
    return params


class State(np.ndarray):
    _desired_shape = (9,)

    def __new__(cls, input_array, dtype=np.bool_):
        return np.asarray(input_array, dtype=dtype).view(cls)

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        assert isinstance(obj, np.ndarray)
        assert obj.shape == self._desired_shape
        assert self.shape == self._desired_shape
        return super().__array_finalize__(obj)

    @property
    def top_left(self):
        return self[0]


class Decision(np.ndarray):
    _desired_shape = (9,)
    _desired_dtype = np.float64

    def __new__(cls, input_array):
        return np.asarray(input_array, dtype=cls._desired_dtype).view(cls)

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        assert isinstance(obj, np.ndarray)
        assert obj.shape == self._desired_shape
        assert self.shape == self._desired_shape
        return super().__array_finalize__(obj)

    @property
    def top_left(self):
        return self[0]


class Player:
    def __init__(self, model: nn.Module):
        self.user_id: str = self.start_game()
        self.done: bool = False
        self.score: int = 0
        self.n_games: int = 0
        self.model: nn.Module = model

    @staticmethod
    def start_game() -> str:
        assert isinstance(user_id := api_client.start_game(), str)

        while not api_client.is_ready(user_id):
            time.sleep(0.1)

        return user_id

    def wait_for_turn(self) -> Optional[int]:
        while True:
            status = api_client.get_status(self.user_id)
            assert status
            _, current_player, your_player_id, winner = status
            if winner is not None:
                self.done = True
                break
            if current_player == your_player_id:
                break
            time.sleep(get_params()["poll_delay"])

        # Delay for human debugging
        time.sleep(get_params()["delay"])

        return winner

    def get_state(self) -> State:
        assert (status := api_client.get_status(self.user_id)) is not None
        board, current_player, your_player_id, winner = status
        if winner is not None:
            self.done = True

        state = [item for sublist in board for item in sublist]

        if your_player_id == 2:
            state = [3 - item for item in state]
            state = [0 if item == 3 else item for item in state]

        return State(state)

    def make_decision(self, state) -> Decision:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            prediction[state > 0] = -100
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return Decision(final_move)

    def move(self, final_move: Decision) -> int:
        row = int(np.argmax(final_move.view(np.ndarray))) // 3
        col = int(np.argmax(final_move.view(np.ndarray))) % 3

        while api_client.make_move(self.user_id, row, col) is not True:
            pass

        # Wait for next player to move
        while True:
            assert (status := api_client.get_status(self.user_id)) is not None
            board, current_player, your_player_id, winner = status
            if winner or current_player == your_player_id:
                break
            time.sleep(0.1)

        # Set the reward based on winner
        if winner is not None:
            self.done = True
            if winner == your_player_id:
                reward = 10
            elif winner == 0:
                reward = 0
            else:
                reward = -10
        else:
            reward = 0

        return reward

    def print_board(self):
        assert (status := api_client.get_status(self.user_id)) is not None
        board, current_player, your_player_id, winner = status
        print()
        for row in board:
            for player in row:
                print(f"{player} ", end="")
            print()

    def play_turn(self) -> Union[tuple[State, State, Decision, int, bool], int]:
        winner = self.wait_for_turn()
        if winner:
            return winner
        state = self.get_state()
        final_move = self.make_decision(state)
        reward = self.move(final_move)
        new_state = self.get_state()
        self.print_board()
        return state, new_state, final_move, reward, self.done


class Agent:
    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(9, 256, 9)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.turns = 0

    def get_state(self, board: api_client.Board, my_player_id: int) -> np.ndarray:
        state = [item for sublist in board for item in sublist]

        if my_player_id == 2:
            state = [3 - item for item in state]
            state = [0 if item == 3 else item for item in state]

        return np.array(state, dtype=int)

    def remember(
        self, state: State, action: Decision, reward: int, next_state: State, done: bool
    ):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(
        self, state: State, action: Decision, reward: int, next_state: State, done: bool
    ):
        self.trainer.train_step(state, action, reward, next_state, done)

    def play_game(self) -> int:
        player = Player(self.model)
        done: bool = False
        score: int = 0
        # while not done:
        while player.wait_for_turn() is None:
            turn = player.play_turn()
            if isinstance(turn, int):
                break
            state, new_state, final_move, reward, done = turn

            score += reward
            self.train_short_memory(state, final_move, reward, new_state, done)

            # Add to long term memory
            self.remember(state, final_move, reward, new_state, done)
        return score

    def train(self, n_games: int):
        plot_progress = PlotProgress()
        record: int = 0
        score: int = 0

        for _ in range(n_games):
            score = self.play_game()
            self.n_games += 1
            self.train_long_memory()

            if score > record:
                record = score
                # self.model.save()

            plot_progress.add_score(score)


class PlotProgress:
    def __init__(self):
        self.plot_scores: list[int] = []
        self.plot_mean_scores: list[float] = []
        self.total_score: int = 0
        self.n_games: int = 0

    def add_score(self, score: int):
        self.n_games += 1
        self.plot_scores.append(score)
        self.total_score += score
        mean_score = self.total_score / self.n_games
        self.plot_mean_scores.append(mean_score)
        plot(self.plot_scores, self.plot_mean_scores)


def train():
    agent = Agent()
    agent.train(10)


if __name__ == "__main__":
    train()
