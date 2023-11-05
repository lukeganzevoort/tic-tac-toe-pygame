from typing import Optional

import requests

base_url = "http://localhost:5000"

Board = list[list[int]]


def start_game() -> Optional[str]:
    response = requests.post(f"{base_url}/start_game")
    if response.status_code == 200:
        data = response.json()
        if not isinstance(user_id := data.get("user_id"), str):
            return None
        return user_id
    else:
        return None


def make_move(user_id: str, row: int, col: int) -> bool:
    payload = {"row": row, "col": col}
    response = requests.post(f"{base_url}/make_move/{user_id}", json=payload)
    if response.status_code != 200:
        return False
    _ = response.json().get("current_player")
    return True


def get_status(user_id: str) -> Optional[tuple[Board, int, int, Optional[int]]]:
    response = requests.get(f"{base_url}/get_status/{user_id}")
    if response.status_code == 200:
        data = response.json()
        if not isinstance(board := data.get("board"), list):
            return None
        if not isinstance(current_player := data.get("current_player"), int):
            return None
        if not isinstance(your_player_id := data.get("your_player_id"), int):
            return None
        if not isinstance(winner := data.get("winner"), (int)) and winner is not None:
            return None
        return (board, current_player, your_player_id, winner)
    else:
        return None


def is_ready(self, user_id) -> Optional[bool]:
    response = requests.get(f"{base_url}/ready/{user_id}")
    if response.status_code == 200:
        data = response.json()
        return data["ready"]
    else:
        return None
