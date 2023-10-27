import sys

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

        # Flask route to make a move via API
        @app.route("/api/move", methods=["POST"])
        def make_move():
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

            self.screen.fill(self.WHITE)
            self.draw_grid()
            self.draw_xo()

            if self.game_over:
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
