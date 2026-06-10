"""``python -m raudio`` entry point (same CLI as the ``raudio`` console script)."""

from raudio.cli import app


def main() -> None:
    app()


if __name__ == "__main__":
    main()
