"""Terminal version of the game (the original text interface, rebuilt on the
new engine). Play against an AI, or watch two AIs play each other.

    python -m battleship_ai play --opponent neural --size 10
    python -m battleship_ai play --watch mcts neural
"""

from __future__ import annotations

import argparse
import time

from .agents import AGENTS, make_agent
from .game import DEFAULT_SIZE, HIT, MAX_SIZE, MIN_SIZE, MISS, SUNK, Board, IllegalMove, Observation

# https://patorjk.com/software/taag/#p=display&h=0&v=0&f=Calvin%20S&t=BATTLESHIP
FONT_ASCII = """
╔╗ ╔═╗╔╦╗╔╦╗╦  ╔═╗╔═╗╦ ╦╦╔═╗
╠╩╗╠═╣ ║  ║ ║  ║╣ ╚═╗╠═╣║╠═╝
╚═╝╩ ╩ ╩  ╩ ╩═╝╚═╝╚═╝╩ ╩╩╩  """

# https://www.asciiart.eu/vehicles/navy
SHIP_ASCII = r"""
                    __/___
              _____/______|
      _______/_____\_______\_____
      \              < < <       |
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

LETTERS = "ABCDEFGHIJKL"
CODE_ICON = {0: "·", MISS: "o", HIT: "X", SUNK: "#"}


def render(title: str, obs: Observation, board: Board | None = None) -> list[str]:
    """Lines for one board. With ``board`` the ships are drawn too."""
    n = obs.size
    ships = {cell for s in board.ships for cell in s.cells} if board else set()
    lines = [title.ljust(2 * n + 3), "   " + " ".join(LETTERS[:n])]
    for r in range(n):
        row = []
        for c in range(n):
            code = int(obs.grid[r, c])
            row.append("■" if code == 0 and (r, c) in ships else CODE_ICON[code])
        lines.append(f"{r + 1:>2} " + " ".join(row))
    return lines


def side_by_side(left: list[str], right: list[str]) -> str:
    width = max(len(s) for s in left) + 6
    return "\n".join(l.ljust(width) + r for l, r in zip(left, right))


def parse_cell(text: str, n: int) -> tuple[int, int]:
    text = text.strip().upper()
    if len(text) < 2 or text[0] not in LETTERS[:n] or not text[1:].isdigit():
        raise ValueError
    r, c = int(text[1:]) - 1, LETTERS.index(text[0])
    if not 0 <= r < n:
        raise ValueError
    return r, c


def describe(who: str, result) -> str:
    cell = f"{LETTERS[result.col]}{result.row + 1}"
    if result.sunk:
        return f"{who} fired {cell}: HIT and SUNK the {result.sunk.name}!"
    return f"{who} fired {cell}: {'HIT' if result.hit else 'miss'}"


def play_human(opponent: str, size: int) -> None:
    agent = make_agent(opponent)
    mine, theirs = Board.random(size), Board.random(size)
    my_view, ai_view = Observation(size), Observation(size)
    print(f"\nYou are playing {agent.name} on a {size}x{size} board. Your fleet was placed at random.")
    print("Enter a cell like B7. Legend: · unknown, o miss, X hit, # sunk, ■ your ship.\n")
    while True:
        print(side_by_side(render("ENEMY WATERS", my_view), render("YOUR FLEET", ai_view, mine)))
        while True:
            try:
                cell = parse_cell(input("\nFire at: "), size)
                result = theirs.fire(*cell)
                break
            except (ValueError, IllegalMove):
                print(f"Enter an unused cell from A1 to {LETTERS[size - 1]}{size}.")
        my_view.apply(result)
        print(describe("You", result))
        if result.game_over:
            print(f"\nVICTORY! You sank the enemy fleet in {len(my_view.history)} shots.")
            return
        result = mine.fire(*agent.select_move(ai_view))
        ai_view.apply(result)
        print(describe(agent.name, result))
        if result.game_over:
            print(side_by_side(render("ENEMY FLEET", my_view, theirs), render("YOUR FLEET", ai_view, mine)))
            print(f"\nDEFEAT. {agent.name} sank your fleet in {len(ai_view.history)} shots.")
            return


def watch(a: str, b: str, size: int, delay: float) -> None:
    agents = [make_agent(a), make_agent(b)]
    boards = [Board.random(size), Board.random(size)]
    views = [Observation(size), Observation(size)]
    turn = 0
    while True:
        result = boards[1 - turn].fire(*agents[turn].select_move(views[turn]))
        views[turn].apply(result)
        print("\n" + side_by_side(render(f"{agents[0].name} ATTACKING", views[0], boards[1]),
                                  render(f"{agents[1].name} ATTACKING", views[1], boards[0])))
        print(describe(agents[turn].name, result))
        if result.game_over:
            print(f"\n{agents[turn].name} wins in {len(views[turn].history)} shots!")
            return
        turn = 1 - turn
        time.sleep(delay)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", choices=list(AGENTS), default=None)
    ap.add_argument("--size", type=int, default=None, choices=range(MIN_SIZE, MAX_SIZE + 1), metavar="N")
    ap.add_argument("--watch", nargs=2, choices=list(AGENTS), metavar=("A", "B"))
    ap.add_argument("--delay", type=float, default=0.3)
    args = ap.parse_args(argv)

    print(FONT_ASCII + SHIP_ASCII + "\n")
    if args.watch:
        return watch(*args.watch, args.size or DEFAULT_SIZE, args.delay)
    opponent = args.opponent
    if opponent is None:
        keys = list(AGENTS)
        for i, key in enumerate(keys, 1):
            print(f"[{i}] {AGENTS[key].name:<9} {AGENTS[key].tagline}")
        while opponent is None:
            choice = input("Choose your opponent: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(keys):
                opponent = keys[int(choice) - 1]
    size = args.size
    while size is None:
        choice = input(f"Board size ({MIN_SIZE}-{MAX_SIZE}, Enter for {DEFAULT_SIZE}): ").strip() or str(DEFAULT_SIZE)
        if choice.isdigit() and MIN_SIZE <= int(choice) <= MAX_SIZE:
            size = int(choice)
    play_human(opponent, size)


if __name__ == "__main__":
    main()
