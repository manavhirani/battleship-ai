import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from battleship_ai.nn_model import WEIGHTS_PATH
from battleship_ai.server import Store, load_meta, make_handler

pytestmark = pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="CNN weights not trained")


@pytest.fixture(scope="module")
def url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Store(), load_meta()))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def call(url, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as res:
            return res.status, json.loads(res.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_static_and_meta(url):
    with urllib.request.urlopen(url + "/") as res:
        assert b"BATTLESHIP" in res.read()
    status, meta = call(url, "/api/meta")
    assert status == 200 and [a["key"] for a in meta["agents"]] == ["random", "mcts", "neural"]


def test_play_a_full_game(url):
    status, fleet = call(url, "/api/random-fleet?size=8")
    assert status == 200 and len(fleet["ships"]) == 5
    status, game = call(url, "/api/play", {"opponent": "random", "size": 8, "ships": fleet["ships"]})
    assert status == 200 and game["turn"] == "player"
    gid = game["id"]
    assert call(url, f"/api/play/{gid}/ai-turn", {})[0] == 409
    cells = [(r, c) for r in range(8) for c in range(8)]
    for r, c in cells:
        status, res = call(url, f"/api/play/{gid}/fire", {"row": r, "col": c})
        assert status == 200
        if res["state"]["turn"] == "over":
            break
        status, ai = call(url, f"/api/play/{gid}/ai-turn", {})
        assert status == 200 and ai["insight"]["heatmap"]
        if ai["state"]["turn"] == "over":
            break
    state = res["state"] if res["state"]["turn"] == "over" else ai["state"]
    assert state["winner"] in ("player", "ai")
    assert state["enemy"]["ships"] is not None  # revealed at the end


def test_errors(url):
    status, game = call(url, "/api/play", {"opponent": "random", "size": 10})
    gid = game["id"]
    assert call(url, f"/api/play/{gid}/fire", {"row": 0, "col": 0})[0] == 200
    assert call(url, f"/api/play/{gid}/ai-turn", {})[0] == 200
    assert call(url, f"/api/play/{gid}/fire", {"row": 0, "col": 0})[0] == 400  # already fired
    assert call(url, "/api/play", {"opponent": "random", "size": 30})[0] == 400
    assert call(url, "/api/play", {"opponent": "nobody"})[0] == 400
    assert call(url, "/api/play/unknown/fire", {"row": 0, "col": 0})[0] == 404
    bad = [{"name": "Carrier", "row": 0, "col": 0, "horizontal": True}]
    assert call(url, "/api/play", {"ships": bad})[0] == 400


def test_duel_steps(url):
    status, game = call(url, "/api/duel", {"a": "random", "b": "neural", "size": 8})
    assert status == 200
    for _ in range(300):
        status, res = call(url, f"/api/duel/{game['id']}/step", {})
        assert status == 200
        if res["state"]["winner"] is not None:
            break
    assert res["state"]["winner"] in (0, 1)
    assert call(url, f"/api/duel/{game['id']}/step", {})[0] == 409


def test_static_files_cannot_escape_web_dir(url):
    import http.client
    host, port = url.removeprefix("http://").split(":")
    conn = http.client.HTTPConnection(host, int(port))
    conn.request("GET", "/../server.py")
    assert conn.getresponse().status == 404
