import time

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.optim as optim


# Define the Tic-Tac-Toe environment
class TicTacToeEnvironment:
    def __init__(self):
        self.base_url = "http://localhost:5000"

    def make_move(self, row: int, col: int):
        response = requests.post(
            self.base_url + "/api/move", json={"row": row, "col": col}
        )

        if response.status_code == 200:
            print("Move successful.")
        else:
            print("Move failed. Check the API endpoint or your move data.")
            print(f"Response: {response.text}")

    def get_board(self) -> list[list[str]]:
        response = requests.get(self.base_url + "/api/board")
        assert response.status_code == 200, "Invalid response from server"
        board_state = response.json()["board"]
        return board_state

    def is_game_over(self) -> bool:
        response = requests.get(self.base_url + "/api/game_over")
        assert response.status_code == 200, "Invalid response from server"
        game_over = response.json()["game_over"]
        return game_over

    def get_state(self) -> list[int]:
        state: list[str] = []
        numeric_board: list[int] = []

        board = self.get_board()

        for row in board:
            state.extend(row)

        for cell in state:
            if cell == "X":
                numeric_board.append(1)
            elif cell == "O":
                numeric_board.append(2)
            else:
                numeric_board.append(0)

        return numeric_board

    def current_player(self) -> str:
        response = requests.get(self.base_url + "/api/current_player")
        assert response.status_code == 200, "Invalid response from server"
        current_player = response.json()["current_player"]
        return current_player

    def reset(self):
        response = requests.post(self.base_url + "/api/reset")

        if response.status_code == 200:
            print("Reset.")
        else:
            print("Reset failed. Check the API endpoint or your move data.")
            print(f"Response: {response.text}")


# Define a Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define a Q-learning agent
class QAgent:
    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.001,
        discount_factor=0.9,
        exploration_prob=0.1,
    ):
        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

    def select_action(self, state):
        print(state)
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(len(state))
        with torch.no_grad():
            print(np.ndarray(state, dtype=int))
            ten = torch.tensor([state], dtype=torch.float)
            print(ten)
            q_values = self.q_network(ten)
            return q_values.argmax().item()

    def train(self, state, action, reward, next_state):
        current_q = self.q_network(torch.tensor(state, dtype=torch.float))[action]
        max_next_q = self.q_network(torch.tensor(next_state, dtype=torch.float)).max()
        target_q = current_q + self.discount_factor * max_next_q

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# Training loop
def train_q_learning_agent():
    env = TicTacToeEnvironment()
    state_size = 9  # 3x3 Tic-Tac-Toe board
    action_size = 9  # 3x3 grid, each cell is a possible action

    agent = QAgent(state_size, action_size)

    num_episodes = 10000
    for episode in range(num_episodes):
        state = env.get_state()
        done = False

        while not done:
            time.sleep(1)
            action = agent.select_action(state)
            env.make_move(action // 3, action % 3)
            next_state = env.get_state()
            reward = 0  # Define your own reward function
            agent.train(state, action, reward, next_state)
            state = next_state

            if env.is_game_over():
                done = True
                env.reset()


# Training the Q-learning agent
train_q_learning_agent()
