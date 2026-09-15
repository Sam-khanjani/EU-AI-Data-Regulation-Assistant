"""Start the chat interface.

    python -m euaia.chat                     http://127.0.0.1:8001
    python -m euaia.chat --host 0.0.0.0      what the Docker image runs

Any other ``chainlit run`` option (``--port``, ``--watch``, ``--debug``) is passed through.
"""

from __future__ import annotations

import os
import sys

from euaia.config import REPO_ROOT

CHAT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # Chainlit reads its settings, translations and public assets from an "app root", which
    # defaults to the working directory and is fixed the moment chainlit is first imported --
    # so both of these must be set before that import, not after.
    os.environ.setdefault("CHAINLIT_APP_ROOT", CHAT_DIR)
    os.environ.setdefault("CHAINLIT_ENV_FILE", str(REPO_ROOT / ".env"))
    from chainlit.cli import cli

    args = sys.argv[1:]
    if "--port" not in args:
        args += ["--port", "8001"]
    cli.main(
        args=["run", os.path.join(CHAT_DIR, "app.py"), "--headless", *args],
        prog_name="python -m euaia.chat",
    )


if __name__ == "__main__":
    main()
