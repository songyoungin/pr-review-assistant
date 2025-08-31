import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from .env if present."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def main() -> None:
    _load_env()
    # Print short confirmation for manual runs
    provider = os.getenv("LLM_PROVIDER", "mock")
    print(f"PR Review Assistant starting (LLM_PROVIDER={provider})")


if __name__ == "__main__":
    main()
