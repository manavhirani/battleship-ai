"""Web server for the Battleship UI (Python standard library only).

    python -m battleship_ai serve --port 8000

JSON API
  GET  /api/meta                  agents, benchmark results, rules
  GET  /api/random-fleet?size=10  a random legal fleet for the player
  POST /api/play                  {opponent, size, ships?} -> new human-vs-AI game
  POST /api/play/<id>/fire        {row, col} -> the player's shot
  POST /api/play/<id>/ai-turn     the AI's reply shot (+ its "insight" heatmap)
  POST /api/duel                  {a, b, size} -> new AI-vs-AI game
  POST /api/duel/<id>/step        one shot of the side whose turn it is
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .agents import AGENTS, Agent, make_agent
from .evaluate import RESULTS_PATH
from .game import (DEFAULT_SIZE, FLEET, MAX_SIZE, MIN_SIZE, Board, IllegalMove,
                   Observation, check_size)
from .nn_model import WEIGHTS_PATH

WEB_DIR = Path(__file__).parent / "web"
MAX_SESSIONS = 500
SESSION_TTL = 3 * 3600


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def side_view(board: Board, obs: Observation, reveal: bool) -> dict:
    """What a viewer sees of ``board``, given the shooter's observation of it."""
    sunk = {s.name for s in obs.sunk_ships}
    return {
        "grid": obs.grid.tolist(),
        "sunk": [s.to_dict() for s in obs.sunk_ships],
        "remaining": [n for n, _ in FLEET if n not in sunk],
        "ships": [dict(s.to_dict(), sunk=s.name in sunk) for s in board.ships] if reveal else None,
        "shots": len(obs.history),
        "hits": sum(r.hit for r in obs.history),
    }


class PlayGame:
    """Human (player) vs an AI agent. The player shoots first."""

    def __init__(self, opponent: str, size: int, ships: Optional[list]) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.touched = time.time()
        self.size = size
        self.agent: Agent = make_agent(opponent)
        self.player_board = Board.from_dicts(size, ships) if ships else Board.random(size)
        self.ai_board = Board.random(size)
        self.player_obs = Observation(size)  # what the player knows of the AI board
        self.ai_obs = Observation(size)      # what the AI knows of the player board
        self.turn = "player"
        self.winner: Optional[str] = None

    def fire(self, row: int, col: int) -> dict:
        if self.turn != "player":
            raise ApiError(HTTPStatus.CONFLICT, "it is not your turn")
        result = self.ai_board.fire(row, col)
        self.player_obs.apply(result)
        if result.game_over:
            self.turn, self.winner = "over", "player"
        else:
            self.turn = "ai"
        return {"shot": result.to_dict(), "state": self.state()}

    def ai_turn(self) -> dict:
        if self.turn != "ai":
            raise ApiError(HTTPStatus.CONFLICT, "it is not the AI's turn")
        move = self.agent.select_move(self.ai_obs)
        result = self.player_board.fire(*move)
        self.ai_obs.apply(result)
        if result.game_over:
            self.turn, self.winner = "over", "ai"
        else:
            self.turn = "player"
        return {"shot": result.to_dict(), "insight": self.agent.insight, "state": self.state()}

    def state(self) -> dict:
        over = self.turn == "over"
        return {
            "id": self.id, "mode": "play", "size": self.size, "turn": self.turn, "winner": self.winner,
            "opponent": {"key": self.agent.key, "name": self.agent.name},
            "enemy": side_view(self.ai_board, self.player_obs, reveal=over),
            "you": side_view(self.player_board, self.ai_obs, reveal=True),
        }


class DuelGame:
    """Two agents play each other; the client steps through the shots."""

    def __init__(self, a: str, b: str, size: int) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.lock = threading.Lock()
        self.touched = time.time()
        self.size = size
        self.agents = [make_agent(a), make_agent(b)]
        self.boards = [Board.random(size), Board.random(size)]
        self.obs = [Observation(size), Observation(size)]  # obs[i]: agent i's view of board 1-i
        self.turn = random.randrange(2)
        self.first = self.turn
        self.winner: Optional[int] = None
        self.insights: list[Optional[dict]] = [None, None]

    def step(self) -> dict:
        if self.winner is not None:
            raise ApiError(HTTPStatus.CONFLICT, "the duel is over")
        i = self.turn
        agent = self.agents[i]
        move = agent.select_move(self.obs[i])
        result = self.boards[1 - i].fire(*move)
        self.obs[i].apply(result)
        self.insights[i] = agent.insight
        if result.game_over:
            self.winner = i
        else:
            self.turn = 1 - i
        return {"shooter": i, "shot": result.to_dict(), "insight": agent.insight, "state": self.state()}

    def state(self) -> dict:
        return {
            "id": self.id, "mode": "duel", "size": self.size, "turn": self.turn,
            "first": self.first, "winner": self.winner,
            "sides": [dict(side_view(self.boards[1 - i], self.obs[i], reveal=True),
                           agent={"key": self.agents[i].key, "name": self.agents[i].name},
                           insight=self.insights[i]) for i in range(2)],
        }


class Store:
    def __init__(self) -> None:
        self.games: dict[str, object] = {}
        self.lock = threading.Lock()

    def add(self, game) -> None:
        with self.lock:
            now = time.time()
            for key in [k for k, g in self.games.items() if now - g.touched > SESSION_TTL]:
                del self.games[key]
            while len(self.games) >= MAX_SESSIONS:
                del self.games[min(self.games, key=lambda k: self.games[k].touched)]
            self.games[game.id] = game

    def get(self, game_id: str, kind: type):
        game = self.games.get(game_id)
        if not isinstance(game, kind):
            raise ApiError(HTTPStatus.NOT_FOUND, "game not found (it may have expired)")
        game.touched = time.time()
        return game


def load_meta() -> dict:
    bench = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None
    training = None
    if WEIGHTS_PATH.exists():
        training = json.loads(str(np.load(WEIGHTS_PATH)["meta"]))
    agents = []
    for key, cls in AGENTS.items():
        stats = bench["agents"].get(key) if bench else None
        agents.append({"key": key, "name": cls.name, "tagline": cls.tagline,
                       "description": cls.description,
                       "benchmark": {k: v for k, v in stats.items() if k != "shots"} if stats else None,
                       "shots": stats["shots"] if stats else None})
    return {
        "agents": agents,
        "fleet": [{"name": n, "length": l} for n, l in FLEET],
        "sizes": {"min": MIN_SIZE, "max": MAX_SIZE, "default": DEFAULT_SIZE},
        "benchmark": {"size": bench["size"], "head_to_head": bench["head_to_head"]} if bench else None,
        "training": training,
    }


def make_handler(store: Store, meta: dict):
    class Handler(BaseHTTPRequestHandler):
        server_version = "BattleshipAI/1.0"

        def log_message(self, fmt, *args):  # keep the console quiet
            pass

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, status: int = 200) -> None:
            self._send(status, json.dumps(data).encode(), "application/json")

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                data = json.loads(self.rfile.read(min(length, 1 << 20)))
            except json.JSONDecodeError:
                raise ApiError(HTTPStatus.BAD_REQUEST, "invalid JSON")
            if not isinstance(data, dict):
                raise ApiError(HTTPStatus.BAD_REQUEST, "expected a JSON object")
            return data

        def _handle(self, method: str) -> None:
            url = urlparse(self.path)
            parts = [p for p in url.path.split("/") if p]
            try:
                if parts[:1] != ["api"]:
                    if method != "GET":
                        raise ApiError(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed")
                    return self._static(url.path)
                self._json(self._api(method, parts[1:], url))
            except ApiError as e:
                self._json({"error": str(e)}, e.status)
            except (IllegalMove, ValueError, KeyError, TypeError) as e:
                self._json({"error": str(e)}, HTTPStatus.BAD_REQUEST)

        def _api(self, method: str, parts: list[str], url):
            route = (method, *parts)
            if route == ("GET", "meta"):
                return meta
            if route == ("GET", "random-fleet"):
                size = check_size(int(parse_qs(url.query).get("size", [DEFAULT_SIZE])[0]))
                return {"ships": [s.to_dict() for s in Board.random(size).ships]}
            if route == ("POST", "play"):
                body = self._body()
                game = PlayGame(str(body.get("opponent", "neural")),
                                check_size(int(body.get("size", DEFAULT_SIZE))), body.get("ships"))
                store.add(game)
                return game.state()
            if route[:2] == ("POST", "play") and len(parts) == 3:
                game = store.get(parts[1], PlayGame)
                with game.lock:
                    if parts[2] == "fire":
                        body = self._body()
                        return game.fire(int(body["row"]), int(body["col"]))
                    if parts[2] == "ai-turn":
                        return game.ai_turn()
            if route == ("POST", "duel"):
                body = self._body()
                game = DuelGame(str(body.get("a", "mcts")), str(body.get("b", "neural")),
                                check_size(int(body.get("size", DEFAULT_SIZE))))
                store.add(game)
                return game.state()
            if route[:2] == ("POST", "duel") and parts[2:] == ["step"]:
                game = store.get(parts[1], DuelGame)
                with game.lock:
                    return game.step()
            raise ApiError(HTTPStatus.NOT_FOUND, "unknown endpoint")

        def _static(self, path: str) -> None:
            target = (WEB_DIR / (path.lstrip("/") or "index.html")).resolve()
            if not target.is_relative_to(WEB_DIR.resolve()) or not target.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype + ("; charset=utf-8" if ctype.startswith("text/") or ctype.endswith("javascript") else ""))

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    if not WEIGHTS_PATH.exists():
        raise SystemExit(f"missing {WEIGHTS_PATH}; run `python -m battleship_ai train` first")
    server = ThreadingHTTPServer((host, port), make_handler(Store(), load_meta()))
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}/"
    print(f"Battleship AI is running at {url}  (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        server.server_close()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Serve the Battleship AI web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--open", action="store_true", help="open the browser")
    args = ap.parse_args(argv)
    serve(args.host, args.port, args.open)


if __name__ == "__main__":
    main()
