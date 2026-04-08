"""OpenEnv validator server entrypoint."""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Run the FastAPI app expected by OpenEnv validators."""

    uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
