import random
import sys
from typing import Optional

import pygame
from flask import Flask, jsonify, request

app = Flask(__name__)


class TicTacToe:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Constants
        self.WIDTH, self.HEIGHT = 400, 400
        self.GRID_SIZE = 3
        self.GRID_WIDTH = self.WIDTH // self.GRID_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.GRID_SIZE
        self.WHITE = (255, 255, 255)
        self.LINE_COLOR = (0, 0, 0)

        # Initialize the screen
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe")

        # Game variables
        self.grid = [["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)]
        self.player_turn = "X"
        self.game_over = False
        self.player_1: Optional[int] = None
        self.player_2: Optional[int] = None

        @app.route("/start_game", methods=["POST"])
        def start_game():
            if self.player_1 is not None and self.player_2 is not None:
                return (
                    jsonify(
                        {
                            "message": "Cannot start game right now,"
                            + "I'm busy with another one."
                        }
                    ),
                    401,
                )
            user_id = random.randint(0, 2**16)
            if self.player_1 is None:
                self.player_1 = user_id
                player_symbol = "X"
            else:
                self.player_2 = user_id
                player_symbol = "0"
                self.player_turn = "X" if random.randint(0, 2) else "O"
                self.game_over = False
                self.grid = [
                    ["" for _ in range(self.GRID_SIZE)] for _ in range(self.GRID_SIZE)
                ]
            return jsonify({"user_id": user_id, "player_symbol": player_symbol})

        # Flask route to make a move via API
        @app.route("/api/move", methods=["POST"])
        def make_move():
            user_id = request.headers.get(
                "X-User-Id"
            )  # Get the user ID from request headers
            if user_id is None or user_id not in [self.player_1, self.player_2]:
                return jsonify({"message": "Invalid or missing user ID"}), 401

            if user_id == self.player_1 and self.player_turn == "O":
                return jsonify({"message": "It's not your turn"})

            if user_id == self.player_2 and self.player_turn == "X":
                return jsonify({"message": "It's not your turn"})

            data = request.get_json()
            row = data["row"]
            col = data["col"]

            if not self.game_over and self.grid[row][col] == "":
                self.grid[row][col] = self.player_turn
                self.player_turn = "O" if self.player_turn == "X" else "X"

                winner = self.check_win()
                if winner:
                    self.game_over = True

                return jsonify({"success": True, "message": "Move successful"})
            else:
                return jsonify({"success": False, "message": "Invalid move"})

        # Flask route to get the current board layout
        @app.route("/api/board", methods=["GET"])
        def get_board():
            return jsonify({"board": self.grid})

        @app.route("/api/game_over", methods=["GET"])
        def get_game_over():
            return jsonify({"game_over": self.game_over})

        @app.route("/api/current_player", methods=["GET"])
        def get_current_player():
            return jsonify({"current_player": self.player_turn})

    # Function to draw the grid
    def draw_grid(self):
        for row in range(1, self.GRID_SIZE):
            pygame.draw.line(
                self.screen,
                self.LINE_COLOR,
                (0, row * self.GRID_HEIGHT),
                (self.WIDTH, row * self.GRID_HEIGHT),
                2,
            )
            pygame.draw.line(
                self.screen,
                self.LINE_COLOR,
                (row * self.GRID_WIDTH, 0),
                (row * self.GRID_WIDTH, self.HEIGHT),
                2,
            )

    # Function to draw X and O on the board
    def draw_xo(self):
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if self.grid[row][col] == "X":
                    pygame.draw.line(
                        self.screen,
                        self.LINE_COLOR,
                        (col * self.GRID_WIDTH, row * self.GRID_HEIGHT),
                        ((col + 1) * self.GRID_WIDTH, (row + 1) * self.GRID_HEIGHT),
                        2,
                    )
                    pygame.draw.line(
                        self.screen,
                        self.LINE_COLOR,
                        ((col + 1) * self.GRID_WIDTH, row * self.GRID_HEIGHT),
                        (col * self.GRID_WIDTH, (row + 1) * self.GRID_HEIGHT),
                        2,
                    )
                elif self.grid[row][col] == "O":
                    pygame.draw.circle(
                        self.screen,
                        self.LINE_COLOR,
                        (
                            col * self.GRID_WIDTH + self.GRID_WIDTH // 2,
                            row * self.GRID_HEIGHT + self.GRID_HEIGHT // 2,
                        ),
                        self.GRID_WIDTH // 2,
                        2,
                    )

    # Function to check for a win
    def check_win(self):
        for row in range(self.GRID_SIZE):
            if self.grid[row][0] == self.grid[row][1] == self.grid[row][2] != "":
                return self.grid[row][0]
        for col in range(self.GRID_SIZE):
            if self.grid[0][col] == self.grid[1][col] == self.grid[2][col] != "":
                return self.grid[0][col]
        if self.grid[0][0] == self.grid[1][1] == self.grid[2][2] != "":
            return self.grid[0][0]
        if self.grid[0][2] == self.grid[1][1] == self.grid[2][0] != "":
            return self.grid[0][2]
        return None

    # Main game loop
    def play(self):
        winner: Optional[str] = None
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if not self.game_over:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        x, y = event.pos
                        col = x // self.GRID_WIDTH
                        row = y // self.GRID_HEIGHT

                        if self.grid[row][col] == "":
                            self.grid[row][col] = self.player_turn
                            self.player_turn = "O" if self.player_turn == "X" else "X"

                        winner = self.check_win()
                        if winner:
                            self.game_over = True
                            self.player_1 = None
                            self.player_2 = None

            self.screen.fill(self.WHITE)
            self.draw_grid()
            self.draw_xo()

            if self.game_over and winner:
                font = pygame.font.Font(None, 36)
                text = font.render(f"Player {winner} wins!", True, self.LINE_COLOR)
                text_rect = text.get_rect(center=(self.WIDTH // 2, self.HEIGHT // 2))
                self.screen.blit(text, text_rect)

            pygame.display.update()


if __name__ == "__main__":
    game = TicTacToe()

    # Run the Flask app in a separate thread
    import threading

    flask_thread = threading.Thread(target=lambda: app.run(host="0.0.0.0", port=5000))
    flask_thread.start()

    # Start the game loop
    game.play()
