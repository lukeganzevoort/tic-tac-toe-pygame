import uuid

from flask import Flask, jsonify, request

from game import TicTacToe

app = Flask(__name)

# Dictionary to store game instances with a unique game_id
games = {}

# Dictionary to store user IDs associated with game IDs
users = {}


@app.route("/start_game", methods=["POST"])
def start_game():
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
def make_move(user_id):
    if user_id not in users:
        return jsonify({"error": "User ID not found"})

    game_id = users.get(user_id)
    game_data = games.get(game_id)

    if game_data is None:
        return jsonify({"error": "Game not found for this user"})

    game = game_data["game"]

    if len(users) % 2 != 0:
        return jsonify({"error": "Waiting for the second player to join"})

    if game.winner() is not None:
        return jsonify({"error": "The game is already over"})

    player = int(request.json.get("player"))
    row = int(request.json.get("row"))
    col = int(request.json.get("col"))

    if player != game.current_player:
        return jsonify({"error": "It's not your turn"})

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


@app.route("/get_board/<user_id>", methods=["GET"])
def get_board(user_id):
    if user_id not in users:
        return jsonify({"error": "User ID not found"})

    game_id = users.get(user_id)
    game_data = games.get(game_id)

    if game_data is None:
        return jsonify({"error": "Game not found for this user"})

    game = game_data["game"]
    board = game.get_board().tolist()
    return jsonify(
        {
            "board": board,
            "player1": game_data["player1"],
            "player2": game_data["player2"],
        }
    )


if __name__ == "__main__":
    app.run()
