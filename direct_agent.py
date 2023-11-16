import random
from collections import deque
from typing import Optional

import numpy as np
import torch

# from recorder import GIFMaker
# from snake_game import Direction, Point, SnakeGameAI
from game import TicTacToe
from helper import PlotProgress
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

Wins = int
Losses = int
Ties = int

PlayerID = int

State = np.ndarray
Move = np.ndarray


class Agent:
    def __init__(self, player_id: PlayerID) -> None:
        self.epsilon = 0  # Randomness
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = Linear_QNet(9, 256, 9)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.plot_progress = PlotProgress(f"Training {player_id}", player_id)

        self.wins: Wins = 0
        self.losses: Losses = 0
        self.ties: Ties = 0

        self.game: Optional[TicTacToe] = None
        self.player: Optional[int] = None
        self.last_state: Optional[State] = None
        self.last_move: Optional[Move] = None
        self.player_id: PlayerID = player_id

    @property
    def record(self) -> tuple[Wins, Losses, Ties]:
        return (self.wins, self.losses, self.ties)

    @property
    def n_games(self):
        return sum(self.record)

    def reset(self):
        self.last_state = None
        self.last_move = None
        self.game = None
        self.player = None

    # def get_state(self) -> np.ndarray:
    #     board = self.game.board.flatten()
    #     if self.player == 2:
    #         board[board > 0] = 3 - board[board > 0]
    #     return np.array(board, dtype=int)

    def get_state_pov(self) -> np.ndarray:
        assert self.game
        assert self.player
        board = self.game.board.flatten()
        if self.player == 1:
            pass
        elif self.player == 2:
            board[board > 0] = 3 - board[board > 0]
        else:
            raise ValueError("Invalid player", self.player)
        return np.array(board, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)

        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state: np.ndarray) -> np.ndarray:
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            prediction = torch.rand(9)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
        prediction[state > 0] = -100
        move = int(torch.argmax(prediction).item())
        final_move[move] = 1

        self.last_state = state
        self.last_move = np.array(final_move)

        return np.array(final_move)

    # def play_turn(self) -> tuple[np.ndarray, np.ndarray]:
    #     state = self.get_state()
    #     final_move = self.get_action(state)

    #     row = int(np.argmax(final_move)) // 3
    #     col = int(np.argmax(final_move)) % 3

    #     assert self.game.move(row, col, self.player)

    #     self.last_state = state
    #     self.last_move = final_move

    #     return state, final_move

    def calculate_reward_and_train(self) -> Optional[int]:
        assert self.last_state is not None
        assert self.last_move is not None
        assert self.game is not None
        new_state = self.get_state_pov()
        winner = self.game.winner()

        done = winner is not None

        # determine reward
        reward = 0
        if done:
            if winner == self.player:
                reward = 10
            elif winner > 0:
                reward = -10

        self.train_short_memory(
            self.last_state, self.last_move, reward, new_state, done
        )
        self.remember(self.last_state, self.last_move, reward, new_state, done)

        if done:
            self.train_long_memory()
            self.plot_progress.add_score(reward)

            if winner == self.player:
                self.wins += 1
                return 1
            elif winner > 0:
                self.losses += 1
                return -1
            else:
                self.ties += 1
                return 0
        return None


# Tournament holds games
# Players get games from tournament
# Players are the API - make a move given X
# Players train based on the result

# This method specifies the AI reacting to the environment...
# The environment reaches out to the AI for a move...
# Or the Agent holds both the environment and the AI and handles the in between


class Agent2:
    def __init__(self, n_players: int):
        self.n_players: int = n_players
        self.players: dict[PlayerID, Agent] = {i: Agent(i) for i in range(n_players)}
        self.games: dict[TicTacToe, tuple[PlayerID, PlayerID]] = {}

    def start_match(self, p1: PlayerID, p2: PlayerID):
        game: TicTacToe = TicTacToe()
        self.games[game] = (p1, p2)
        self.players[p1].game = game
        self.players[p1].player = 1
        self.players[p2].game = game
        self.players[p2].player = 2

    def play_turns(self):
        player: Agent
        state: State
        move: Move

        print(self.games)

        # Play a turn
        for game, (p1, p2) in self.games.items():
            if game.current_player == 0:
                print("Game finished")
                continue
            elif game.current_player == 1:
                player = self.players[p1]
            elif game.current_player == 2:
                player = self.players[p2]
            else:
                raise ValueError("Invalid current player", game.current_player)

            state = player.get_state_pov()

            if player.last_state is not None and player.last_move is not None:
                print("Training")
                player.calculate_reward_and_train()

            move = player.get_action(state)

            row = int(np.argmax(move)) // 3
            col = int(np.argmax(move)) % 3

            assert game.move(row, col, game.current_player)

        # Clean up game list
        for game in list(self.games.keys()):
            if game.current_player == 0:
                p1, p2 = self.games.pop(game)
                assert (game.winner()) is not None
                self.players[p1].calculate_reward_and_train()
                self.players[p1].reset()
                self.players[p2].calculate_reward_and_train()
                self.players[p2].reset()


def main():
    agent = Agent2(8)

    while True:
        agent.start_match(0, 1)
        agent.start_match(2, 3)
        agent.start_match(4, 5)
        agent.start_match(6, 7)

        while len(agent.games) > 0:
            print("Play next turn")
            agent.play_turns()


if __name__ == "__main__":
    main()
