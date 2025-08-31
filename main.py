import os
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from .env if present."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def main() -> None:
    """Entry point for running the MVP orchestrator pipeline.

    This function loads environment variables, prints a short status
    line and executes the MVP orchestrator end-to-end using a local
    sample PR identifier. The final report is printed and written to
    `final_report.json` in the repository root.
    """

    _load_env()

    provider = os.getenv("LLM_PROVIDER", "mock")
    print(f"PR Review Assistant starting (LLM_PROVIDER={provider})")

    # Run the MVP orchestrator pipeline
    try:
        # Import here to keep startup lightweight when main isn't used
        from agents.planner import MVPOrchestrator

        async def _run() -> None:
            orch = MVPOrchestrator()
            state = await orch.initialize_workflow("local-sample-pr")
            state = await orch.run_mvp_pipeline(state)

            # Print and persist final report
            final = state.outputs.final_report or "{}"
            print(final)
            Path("final_report.json").write_text(final, encoding="utf-8")

        import asyncio

        asyncio.run(_run())
    except Exception as e:
        print(f"Failed to run MVP orchestrator: {e}")


if __name__ == "__main__":
    main()
