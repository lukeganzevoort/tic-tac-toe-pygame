import logging
from typing import Optional

import requests

base_url = "http://localhost:5000"

Board = list[list[int]]
# Configure the logging settings
logging.basicConfig(level=logging.WARNING)


def start_game() -> Optional[str]:
    response = requests.post(f"{base_url}/start_game")
    if response.status_code == 200:
        data = response.json()
        if not isinstance(user_id := data.get("user_id"), str):
            logging.warning("Invalid 'user_id' received from the API")
            return None
        return user_id
    else:
        logging.warning(f"Failed to start a game. Status code: {response.status_code}")
        return None


def setup_game() -> Optional[tuple[str, str]]:
    response = requests.post(f"{base_url}/setup_game")
    if response.status_code == 200:
        data = response.json()
        if not isinstance(player1 := data.get("player1"), str):
            logging.warning("Invalid 'player1' received from the API")
            return None
        if not isinstance(player2 := data.get("player2"), str):
            logging.warning("Invalid 'player2' received from the API")
            return None
        return player1, player2
    else:
        logging.warning(f"Failed to setup a game. Status code: {response.status_code}")
        return None


def make_move(user_id: str, row: int, col: int) -> Optional[bool]:
    payload = {"row": row, "col": col}
    response = requests.post(f"{base_url}/make_move/{user_id}", json=payload)
    if response.status_code == 200:
        _ = response.json().get("current_player")
        return True
    else:
        if response.json().get("error") == "Invalid move. Cell is already occupied.":
            return False
        logging.warning(
            f"Failed to make a move. Status code: {response.status_code}"
            + f"{response.json()}"
        )
        return None


def get_status(user_id: str) -> Optional[tuple[Board, int, int, Optional[int]]]:
    response = requests.get(f"{base_url}/get_status/{user_id}")
    if response.status_code == 200:
        data = response.json()
        board = data.get("board")
        current_player = data.get("current_player")
        your_player_id = data.get("your_player_id")
        winner = data.get("winner")

        if (
            not isinstance(board, list)
            or not isinstance(current_player, int)
            or not isinstance(your_player_id, int)
            or (winner is not None and not isinstance(winner, int))
        ):
            logging.warning("Invalid data received from the API")
            return None

        return board, current_player, your_player_id, winner
    else:
        logging.warning(
            f"Failed to get game status. Status code: {response.status_code}"
        )
        return None


def is_ready(user_id: str) -> Optional[bool]:
    response = requests.get(f"{base_url}/ready/{user_id}")
    if response.status_code == 200:
        data = response.json()
        return data["ready"]
    else:
        logging.warning(
            f"Failed to check readiness. Status code: {response.status_code}"
        )
        return None
