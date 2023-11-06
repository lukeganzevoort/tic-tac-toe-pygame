import json
import random
import time
from collections import deque

import numpy as np
import torch

import api_client
from helper import plot
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


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
        assert all(state) >= 0 and all(state) < 3
        return np.array(state, dtype=int)

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

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 180 - self.n_games + 20
        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 8)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = int(torch.argmax(prediction).item())
            final_move[move] = 1

        return final_move

    # def play_step(self, user_id: str, row: int, col: int) -> int:
    #     success = api_client.make_move(user_id, row, col)
    #     assert success is not None
    #     if not success:
    #         reward = -1
    #     else:
    #         reward = 1

    #     return reward


def get_params():
    with open("params.json", "r") as f:
        params = json.load(f)
    return params


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    while True:
        assert isinstance(user_id := api_client.start_game(), str)
        agent.turns = 0
        done = False
        score = 0
        while not api_client.is_ready(user_id):
            time.sleep(0.1)

        while not done:
            while True:
                status = api_client.get_status(user_id)
                assert status
                _, current_player, your_player_id, winner = status
                if winner is not None:
                    done = True
                    break
                if current_player == your_player_id:
                    break
                time.sleep(get_params()["poll_delay"])

            time.sleep(get_params()["delay"])
            reward = 0

            # get old state
            assert (status := api_client.get_status(user_id)) is not None
            board, current_player, your_player_id, winner = status
            state_old = agent.get_state(board, your_player_id)

            final_move = agent.get_action(state_old)
            print(final_move)
            row = final_move.index(1) // 3
            col = final_move.index(1) % 3

            if not done:
                reward = api_client.make_move(user_id, row, col)
                assert isinstance(reward, bool)
                if reward:
                    reward = 1
                else:
                    reward = -1

            assert (status := api_client.get_status(user_id)) is not None
            board, current_player, your_player_id, winner = status
            if winner is not None:
                done = True
                if winner == your_player_id:
                    reward = 10
                elif winner == 0:
                    reward = 0
                else:
                    reward = -10

            state_new = agent.get_state(board, your_player_id)

            print("reward", reward)
            score += reward
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                agent.n_games += 1
                agent.train_long_memory()

                # agent.model.save()
                if score > record:
                    record = score
                    # agent.model.save()

                print("Game", agent.n_games, "score", score, "Record:", record)

                plot_scores.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

            print()
            for row in board:
                for player in row:
                    print(f"{player} ", end="")
                print()


if __name__ == "__main__":
    train()
