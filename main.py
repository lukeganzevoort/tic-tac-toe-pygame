import sys

import pygame

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 3
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
WHITE = (255, 255, 255)
LINE_COLOR = (0, 0, 0)

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic-Tac-Toe")

# Game variables
grid = [['' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
player_turn = 'X'
game_over = False

# Function to draw the grid
def draw_grid():
    for row in range(1, GRID_SIZE):
        pygame.draw.line(screen, LINE_COLOR, (0, row * GRID_HEIGHT), (WIDTH, row * GRID_HEIGHT), 2)
        pygame.draw.line(screen, LINE_COLOR, (row * GRID_WIDTH, 0), (row * GRID_WIDTH, HEIGHT), 2)

# Function to draw X and O on the board
def draw_xo():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if grid[row][col] == 'X':
                pygame.draw.line(screen, LINE_COLOR, (col * GRID_WIDTH, row * GRID_HEIGHT),
                                 ((col + 1) * GRID_WIDTH, (row + 1) * GRID_HEIGHT), 2)
                pygame.draw.line(screen, LINE_COLOR, ((col + 1) * GRID_WIDTH, row * GRID_HEIGHT),
                                 (col * GRID_WIDTH, (row + 1) * GRID_HEIGHT), 2)
            elif grid[row][col] == 'O':
                pygame.draw.circle(screen, LINE_COLOR, (col * GRID_WIDTH + GRID_WIDTH // 2, row * GRID_HEIGHT + GRID_HEIGHT // 2), GRID_WIDTH // 2, 2)

# Function to check for a win
def check_win():
    for row in range(GRID_SIZE):
        if grid[row][0] == grid[row][1] == grid[row][2] != '':
            return grid[row][0]
    for col in range(GRID_SIZE):
        if grid[0][col] == grid[1][col] == grid[2][col] != '':
            return grid[0][col]
    if grid[0][0] == grid[1][1] == grid[2][2] != '':
        return grid[0][0]
    if grid[0][2] == grid[1][1] == grid[2][0] != '':
        return grid[0][2]
    return None

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if not game_over:
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                col = x // GRID_WIDTH
                row = y // GRID_HEIGHT

                if grid[row][col] == '':
                    grid[row][col] = player_turn
                    player_turn = 'O' if player_turn == 'X' else 'X'

                winner = check_win()
                if winner:
                    game_over = True

    screen.fill(WHITE)
    draw_grid()
    draw_xo()

    if game_over:
        font = pygame.font.Font(None, 36)
        text = font.render(f"Player {winner} wins!", True, LINE_COLOR)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(text, text_rect)

    pygame.display.update()
