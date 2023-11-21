import random
from collections import deque
from typing import Optional

import numpy as np
import pygame
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
RoundMatches = list[tuple[PlayerID, PlayerID]]

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

    @property
    def win_rate(self) -> float:
        if self.n_games == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.n_games

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
            # self.plot_progress.add_score(reward)

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


class UI(Agent):
    def __init__(self, player_id: int):
        pygame.init()

        # Define the screen dimensions
        self.SCREEN_WIDTH, self.SCREEN_HEIGHT = 600, 600
        self.SCREEN_SIZE = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)

        # Colors
        self.WHITE = (255, 255, 255)
        self.LINE_COLOR = (0, 0, 0)

        # Initialize the screen
        self.screen = pygame.display.set_mode(self.SCREEN_SIZE)
        pygame.display.set_caption("Tic-Tac-Toe")

        # Font
        self.font = pygame.font.Font(None, 120)

        # Game variables
        self.running = True
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.game: Optional[TicTacToe] = None
        self.player: Optional[int] = None

        super().__init__(player_id)

    # Function to draw the grid
    def draw_grid(self):
        cell_width = self.SCREEN_WIDTH // 3
        cell_height = self.SCREEN_HEIGHT // 3

        # Vertical lines
        for x in range(1, 3):
            pygame.draw.line(
                self.screen,
                self.LINE_COLOR,
                (cell_width * x, 0),
                (cell_width * x, self.SCREEN_HEIGHT),
                2,
            )

        # Horizontal lines
        for y in range(1, 3):
            pygame.draw.line(
                self.screen,
                self.LINE_COLOR,
                (0, cell_height * y),
                (self.SCREEN_WIDTH, cell_height * y),
                2,
            )

    # Function to draw X and O
    def draw_x_o(self):
        cell_width = self.SCREEN_WIDTH // 3
        cell_height = self.SCREEN_HEIGHT // 3

        for row in range(3):
            for col in range(3):
                if self.board[row][col] == 1:
                    x_text = self.font.render("X", True, self.LINE_COLOR)
                    self.screen.blit(
                        x_text, (col * cell_width + cell_width // 3, row * cell_height)
                    )
                elif self.board[row][col] == 2:
                    o_text = self.font.render("O", True, self.LINE_COLOR)
                    self.screen.blit(
                        o_text, (col * cell_width + cell_width // 3, row * cell_height)
                    )

    # Function to make a move using the API client
    def make_move(self, row, col):
        assert self.game
        assert self.player
        assert self.game.move(row, col, self.player)

    # Function to update the board from the API using the API client
    def update_board(self):
        assert self.game
        self.board = self.game.board.tolist()

    # Main game loop
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the clicked cell
                    x, y = event.pos
                    col = x // (self.SCREEN_WIDTH // 3)
                    row = y // (self.SCREEN_HEIGHT // 3)

                    # Check if the cell is empty
                    if self.board[row][col] == 0:
                        self.make_move(row, col)

            self.update_board()  # Update the board from the API

            # Clear the screen
            self.screen.fill(self.WHITE)

            # Draw the grid
            self.draw_grid()

            # Draw X and O
            self.draw_x_o()

            # Update the display
            pygame.display.update()

        pygame.quit()

    def reset(self):
        self.last_state = None
        self.last_move = None
        self.game = None
        self.player = None

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
        pass

    def train_long_memory(self):
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def get_action(self, state: np.ndarray) -> np.ndarray:
        running: bool = True
        col, row = 0, 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Get the clicked cell
                    x, y = event.pos
                    col = x // (self.SCREEN_WIDTH // 3)
                    row = y // (self.SCREEN_HEIGHT // 3)

                    # Check if the cell is empty
                    if self.board[row][col] == 0:
                        # self.make_move(row, col)
                        running = False

            self.update_board()  # Update the board from the API

            # Clear the screen
            self.screen.fill(self.WHITE)

            # Draw the grid
            self.draw_grid()

            # Draw X and O
            self.draw_x_o()

            # Update the display
            pygame.display.update()

        # pygame.quit()

        final_move = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        move = row * 3 + col
        final_move[move] = 1

        return np.array(final_move)

    def calculate_reward_and_train(self) -> Optional[int]:
        assert self.game is not None

        winner = self.game.winner()
        done = winner is not None

        if done:
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

    def round_robin_schedule(self) -> list[RoundMatches]:
        """Generates a round-robin schedule for a list of players."""
        player_ids: list[PlayerID] = list(self.players.keys())
        schedule: list[RoundMatches] = []

        if len(player_ids) % 2:
            # If the number of players is odd, add a dummy player for bye rounds
            player_ids.append(-1)

        for i in range(len(player_ids) - 1):
            round_matches: RoundMatches = []
            for j in range(len(player_ids) // 2):
                round_matches.append((player_ids[j], player_ids[-j - 1]))
            player_ids.insert(1, player_ids.pop())  # Rotate the list of players
            schedule.append(round_matches)

        return schedule

    def run_round(self, round_matches: RoundMatches):
        for match in round_matches:
            if -1 in match:  # This is a bye
                continue
            self.start_match(*match)

        while len(self.games) > 0:
            self.play_turns()

    def play_turns(self):
        player: Agent
        state: State
        move: Move

        # Play a turn
        for game, (p1, p2) in self.games.items():
            if game.current_player == 0:
                continue
            elif game.current_player == 1:
                player = self.players[p1]
            elif game.current_player == 2:
                player = self.players[p2]
            else:
                raise ValueError("Invalid current player", game.current_player)

            state = player.get_state_pov()

            if player.last_state is not None and player.last_move is not None:
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

    def get_leader_board(self) -> list[Agent]:
        return list(
            sorted(self.players.values(), key=lambda x: x.win_rate, reverse=True)
        )

    def print_leader_board(self):
        print("PlayerID   PCT  Wins  Losses  Ties")
        for player in self.get_leader_board():
            print(
                f"{player.player_id:8}"
                + f"  {player.win_rate:.2f}"
                + f"{player.wins:6}"
                + f"{player.losses:8}"
                + f"{player.ties:6}"
            )


def main():
    agent = Agent2(32)
    schedule = agent.round_robin_schedule()

    for _ in range(10):
        random.shuffle(schedule)

        for rnd in schedule:
            agent.run_round(rnd)

        agent.print_leader_board()

    top_agent = agent.get_leader_board()[0]
    print(top_agent.win_rate)
    agent.players[-1] = UI(-1)
    while True:
        agent.start_match(top_agent.player_id, -1)
        while len(agent.games) > 0:
            agent.play_turns()

    return


# Example usage:
# players = ["A", "B", "C", "D", "E"]
# schedule = round_robin(players)

# for round_number, matches in enumerate(schedule, 1):
#     print(f"Round {round_number}:")
#     for match in matches:
#         print(f"{match[0]} vs {match[1]}")


if __name__ == "__main__":
    main()
