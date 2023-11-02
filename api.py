import uuid
from typing import Dict, Union

from flask import Flask, Response, jsonify, request

from game import TicTacToe

app = Flask(__name__)

# Dictionary to store game instances with a unique game_id
games: Dict[str, Dict[str, Union[TicTacToe, None, str]]] = {}

# Dictionary to store user IDs associated with game IDs
users: Dict[str, str] = {}


@app.route("/start_game", methods=["POST"])
def start_game() -> Response:
    if len(users) % 2 == 0:
        game_id = str(uuid.uuid4())
        game = TicTacToe()
        games[game_id] = {"game": game, "player1": None, "player2": None}
    else:
        game_id = list(games.keys())[-1]
    user_id = str(uuid.uuid4())
    users[user_id] = game_id
    if games[game_id]["player1"] is None:
        games[game_id]["player1"] = user_id
    else:
        games[game_id]["player2"] = user_id
    return jsonify({"game_id": game_id, "user_id": user_id})


@app.route("/make_move/<user_id>", methods=["POST"])
def make_move(user_id: str) -> Response:
    if user_id not in users:
        return jsonify({"error": "User ID not found"})

    if (game_id := users.get(user_id)) is None:
        return jsonify({"error": "User ID not assigned to game"})

    if (game_data := games.get(game_id)) is None:
        return jsonify({"error": "Game not found for this user"})

    if not isinstance(game := game_data["game"], TicTacToe):
        return jsonify({"error": "Game not set correctly."})

    if game.winner() is not None:
        return jsonify({"error": "The game is already over"})

    if not isinstance(player1 := game_data.get("player1"), str):
        return jsonify({"error": "Waiting for the first player to join"})

    if not isinstance(player2 := game_data.get("player2"), str):
        return jsonify({"error": "Waiting for the second player to join"})

    if user_id == player1:
        player = 1
    elif user_id == player2:
        player = 2
    else:
        return jsonify({"error": "user_id is not part of this game"})

    if player != game.current_player:
        return jsonify({"error": "It's not your turn"})

    if not isinstance(req_json := request.get_json(), dict):
        return jsonify({"error": "row and col not specified"})

    if not isinstance(row := req_json.get("row"), Union[int, float]):
        return jsonify({"error": "row must be a number"})
    row = int(row)

    if not isinstance(col := req_json.get("col"), Union[int, float]):
        return jsonify({"error": "col must be a number"})
    col = int(col)

    if not (0 <= row < 3 and 0 <= col < 3):
        return jsonify(
            {"error": "Invalid move. Row and column should be in the range [0, 2]"}
        )

    if game.move(row, col):
        return jsonify(
            {"message": "Move successful", "current_player": game.current_player}
        )
    else:
        return jsonify(
            {"error": "Invalid move. Cell is already occupied or the game is over"}
        )


@app.route("/get_status/<user_id>", methods=["GET"])
def get_status(user_id: str) -> Response:
    if user_id not in users:
        return jsonify({"error": "User ID not found"})

    if (game_id := users.get(user_id)) is None:
        return jsonify({"error": "User ID not assigned to game"})

    if (game_data := games.get(game_id)) is None:
        return jsonify({"error": "Game not found for this user"})

    if not isinstance(game := game_data["game"], TicTacToe):
        return jsonify({"error": "Game not set correctly."}, 501)

    board = game.get_board().tolist()

    if not isinstance(player1 := game_data.get("player1"), str):
        return jsonify({"error": "Waiting for the first player to join"})

    if not isinstance(player2 := game_data.get("player2"), str):
        return jsonify({"error": "Waiting for the second player to join"})

    if user_id == player1:
        player = 1
    elif user_id == player2:
        player = 2
    else:
        return jsonify({"error": "user_id is not part of this game"})

    winner = game.winner()

    return jsonify(
        {
            "board": board,
            "current_player": game.current_player,
            "your_player_id": player,
            "winner": winner,
        }
    )


if __name__ == "__main__":
    app.run()
