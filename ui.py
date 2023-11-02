import time
from typing import Optional

import pygame
import requests


class UI:
    def __init__(self):
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
        self.player = 1  # Assume the player is X
        self.board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.user_id = self.start_game()

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

    # Function to make a move using the API
    def make_move(self, row, col):
        response = requests.post(
            f"http://localhost:5000/make_move/{self.user_id}",
            json={"row": row, "col": col},
        )
        if response.status_code == 200:
            self.board[row][col] = self.player
            self.player = 3 - self.player  # Switch player
        else:
            print(response.json(), response.status_code)

    # Function to update the board from the API
    def update_board(self):
        response = requests.get(f"http://localhost:5000/get_status/{self.user_id}")
        if response.status_code == 200:
            data = response.json()
            self.board = data["board"]
            self.player = data["current_player"]
        else:
            print(response.json(), response.status_code)

    # Function to start the game and get the user_id
    def start_game(self) -> Optional[str]:
        response = requests.post("http://localhost:5000/start_game")
        if response.status_code == 200:
            data = response.json()
            return data["user_id"]
        else:
            print(response.json(), response.status_code)
            return None

    def is_ready(self) -> bool:
        response = requests.get(f"http://localhost:5000/ready/{self.user_id}")
        if response.status_code == 200:
            data = response.json()
            return data["ready"]
        else:
            print(response.json(), response.status_code)
            return False

    # Main game loop
    def run(self):
        while not self.is_ready():
            time.sleep(0.1)

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


if __name__ == "__main__":
    ui = UI()
    ui.run()
