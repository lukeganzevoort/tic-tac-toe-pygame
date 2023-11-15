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
def start_game() -> tuple[Response, int]:
    if len(games) == 0 or games[list(games.keys())[-1]].get("player2"):
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
    return jsonify({"game_id": game_id, "user_id": user_id}), 200


@app.route("/setup_game", methods=["POST"])
def setup_game() -> tuple[Response, int]:
    game_id = str(uuid.uuid4())
    player_1 = str(uuid.uuid4())
    player_2 = str(uuid.uuid4())

    games[game_id] = {"game": TicTacToe(), "player1": player_1, "player2": player_2}
    users[player_1] = game_id
    users[player_2] = game_id

    return jsonify({"game_id": game_id, "player1": player_1, "player2": player_2}), 200


@app.route("/make_move/<user_id>", methods=["POST"])
def make_move(user_id: str) -> tuple[Response, int]:
    if user_id not in users:
        return jsonify({"error": "User ID not found"}), 401

    if (game_id := users.get(user_id)) is None:
        return jsonify({"error": "User ID not assigned to game"}), 500

    if (game_data := games.get(game_id)) is None:
        return jsonify({"error": "Game not found for this user"}), 500

    if not isinstance(game := game_data["game"], TicTacToe):
        return jsonify({"error": "Game not set correctly."}), 500

    if game.winner() is not None:
        return jsonify({"error": "The game is already over"}), 400

    if not isinstance(player1 := game_data.get("player1"), str):
        return jsonify({"error": "Waiting for the first player to join"}), 503

    if not isinstance(player2 := game_data.get("player2"), str):
        return jsonify({"error": "Waiting for the second player to join"}), 503

    if user_id == player1:
        player = 1
    elif user_id == player2:
        player = 2
    else:
        return jsonify({"error": "user_id is not part of this game"}), 500

    if player != game.current_player:
        return jsonify({"error": "It's not your turn"}), 503

    if not isinstance(req_json := request.get_json(), dict):
        return jsonify({"error": "row and col not specified"}), 400

    if not isinstance(row := req_json.get("row"), Union[int, float]):
        return jsonify({"error": "row must be a number"}), 400
    row = int(row)

    if not isinstance(col := req_json.get("col"), Union[int, float]):
        return jsonify({"error": "col must be a number"}), 400
    col = int(col)

    if not (0 <= row < 3 and 0 <= col < 3):
        return (
            jsonify(
                {"error": "Invalid move. Row and column should be in the range [0, 2]"}
            ),
            400,
        )

    if game.move(row, col):
        return (
            jsonify(
                {"message": "Move successful", "current_player": game.current_player}
            ),
            200,
        )
    else:
        return (
            jsonify({"error": "Invalid move. Cell is already occupied."}),
            400,
        )


@app.route("/get_status/<user_id>", methods=["GET"])
def get_status(user_id: str) -> tuple[Response, int]:
    if user_id not in users:
        return jsonify({"error": "User ID not found"}), 401

    if (game_id := users.get(user_id)) is None:
        return jsonify({"error": "User ID not assigned to game"}), 500

    if (game_data := games.get(game_id)) is None:
        return jsonify({"error": "Game not found for this user"}), 500

    if not isinstance(game := game_data["game"], TicTacToe):
        return jsonify({"error": "Game not set correctly."}), 500

    board = game.get_board().tolist()

    if not isinstance(player1 := game_data.get("player1"), str):
        return jsonify({"error": "Waiting for the first player to join"}), 503

    if not isinstance(player2 := game_data.get("player2"), str):
        return jsonify({"error": "Waiting for the second player to join"}), 503

    if user_id == player1:
        player = 1
    elif user_id == player2:
        player = 2
    else:
        return jsonify({"error": "user_id is not part of this game"}), 500

    winner = game.winner()

    return (
        jsonify(
            {
                "board": board,
                "current_player": game.current_player,
                "your_player_id": player,
                "winner": winner,
            }
        ),
        200,
    )


@app.route("/ready/<user_id>", methods=["GET"])
def ready(user_id: str) -> tuple[Response, int]:
    if user_id not in users:
        return jsonify({"error": "User ID not found"}), 401

    if (game_id := users.get(user_id)) is None:
        return jsonify({"error": "User ID not assigned to game"}), 500

    if (game_data := games.get(game_id)) is None:
        return jsonify({"error": "Game not found for this user"}), 500

    player1 = game_data.get("player1")
    player2 = game_data.get("player2")

    return (jsonify({"ready": player1 and player2}), 200)


if __name__ == "__main__":
    app.run()
