# Tic Tac Toe Pygame

This is a really basic implementation of Tic Tac Toe using pygame.
The game was implemented with an API to be able to make moves with POST requests.

## Playing via API

Terminal Example:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"row": 0, "col": 0}' http://localhost:5000/api/move
```

To make a move POST to `http://localhost:5000/api/move` with the "row" and "col" number.
To get board make a GET request to `http://localhost:5000/api/board`