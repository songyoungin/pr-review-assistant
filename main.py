import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_env() -> None:
    """Load environment variables from .env if present."""
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)


def _generate_diff_and_files_from_commit(repo_path: str, commit: str) -> None:
    """Generate `local-sample-pr.diff` and `local-sample-pr.files` from a commit.

    Args:
        repo_path: Path to the git repository.
        commit: Commit hash or ref to show.
    """
    from tools.git.git_changes import GitChangeDetector

    detector = GitChangeDetector(repo_path)
    if not detector.repo:
        raise RuntimeError("Git repository not initialized")

    # Get unified diff for the commit
    diff_text = detector.repo.git.show(commit)
    Path("local-sample-pr.diff").write_text(diff_text, encoding="utf-8")

    # Get file list for the commit
    files_text = detector.repo.git.show("--name-only", "--pretty=", commit)
    Path("local-sample-pr.files").write_text(files_text, encoding="utf-8")


def _generate_diff_and_files_from_range(repo_path: str, base: str, head: str) -> None:
    """Generate `local-sample-pr.diff` and `local-sample-pr.files` from base..head.

    Args:
        repo_path: Path to the git repository.
        base: Base ref (e.g. main).
        head: Head ref (e.g. feature branch).
    """
    from tools.git.git_changes import GitChangeDetector

    detector = GitChangeDetector(repo_path)
    if not detector.repo:
        raise RuntimeError("Git repository not initialized")

    # Unified diff between base and head
    diff_text = detector.repo.git.diff(f"{base}..{head}")
    Path("local-sample-pr.diff").write_text(diff_text, encoding="utf-8")

    # File path list between base and head
    files_text = detector.repo.git.diff("--name-only", f"{base}..{head}")
    Path("local-sample-pr.files").write_text(files_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """Entry point for running the MVP orchestrator pipeline with optional Git-based inputs.

    This function supports generating the required `.diff` and `.files` inputs from a
    commit or a branch range (`--base`/`--head`) and then runs the MVP orchestrator.

    Args:
        argv: Optional list of CLI arguments (for testing). If None, sys.argv is used.
    """

    _load_env()

    parser = argparse.ArgumentParser(
        description="Run MVP orchestrator with optional git inputs"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--commit", help="Commit hash or ref to generate diff/files from"
    )
    group.add_argument(
        "--range",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help="Base and head refs to diff (base head)",
    )
    parser.add_argument(
        "--repo-path", default=".", help="Path to git repository (default: current dir)"
    )
    args = parser.parse_args(argv)

    provider = os.getenv("LLM_PROVIDER", "mock")
    print(f"PR Review Assistant starting (LLM_PROVIDER={provider})")

    try:
        # Generate input files if requested
        if args.commit:
            print(f"Generating diff/files from commit: {args.commit}")
            _generate_diff_and_files_from_commit(args.repo_path, args.commit)
        elif args.range:
            base, head = args.range
            print(f"Generating diff/files from range: {base}..{head}")
            _generate_diff_and_files_from_range(args.repo_path, base, head)

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
        sys.exit(1)


if __name__ == "__main__":
    main()
