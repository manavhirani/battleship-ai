"""python -m battleship_ai [serve|play|evaluate|train] [options]

serve     start the web UI (default)
play      play in the terminal, or watch two AIs (--watch A B)
evaluate  benchmark the agents
train     train the CNN agent (needs PyTorch)
"""

import sys


def main() -> None:
    args = sys.argv[1:]
    command = args.pop(0) if args and not args[0].startswith("-") else "serve"
    if command == "serve":
        from .server import main as run
    elif command == "play":
        from .cli import main as run
    elif command == "evaluate":
        from .evaluate import main as run
    elif command == "train":
        from .train import main as run
    else:
        print(__doc__)
        raise SystemExit(f"unknown command {command!r}")
    run(args)


if __name__ == "__main__":
    main()
