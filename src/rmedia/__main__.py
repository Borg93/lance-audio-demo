"""``python -m raudio`` entry point (same CLI as the ``raudio`` console script)."""

import sys

from rmedia.cli import app
from rmedia.errors import RaudioError


def main() -> None:
    try:
        app()
    except RaudioError as e:
        # Library code raises domain errors; the entry point owns the exit.
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
