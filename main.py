import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger


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


def _generate_diff_and_files_from_range(
    repo_path: str, base: str, head: str, pr_url: str | None = None
) -> None:
    """Generate `local-sample-pr.diff` and `local-sample-pr.files` from base..head.

    Args:
        repo_path: Path to the git repository.
        base: Base ref (e.g. main).
        head: Head ref (e.g. feature branch).
    """
    import re

    from tools.git.git_changes import GitChangeDetector

    detector = GitChangeDetector(repo_path)
    if not detector.repo:
        raise RuntimeError("Git repository not initialized")

    diff_text = ""

    # Primary attempt: local base..head
    try:
        diff_text = detector.repo.git.diff(f"{base}..{head}")
    except Exception:
        # Try to fetch remote refs and use origin/base..origin/head
        try:
            detector.repo.git.fetch("origin", "--prune")
        except Exception:
            pass

        try:
            diff_text = detector.repo.git.diff(f"origin/{base}..origin/{head}")
        except Exception:
            diff_text = ""

    # If still empty and pr_url provided, try fetching PR head refs
    if (not diff_text) and pr_url:
        m = re.search(r"/pull/(\d+)", pr_url)
        if m:
            pr_num = m.group(1)
            tmp_branch = f"pr-{pr_num}-head"
            try:
                # Attempt to fetch the PR head as a local branch
                detector.repo.git.fetch(
                    "origin", f"refs/pull/{pr_num}/head:{tmp_branch}"
                )
                try:
                    diff_text = detector.repo.git.diff(f"{base}..{tmp_branch}")
                except Exception:
                    try:
                        diff_text = detector.repo.git.diff(
                            f"origin/{base}..{tmp_branch}"
                        )
                    except Exception:
                        diff_text = ""
            except Exception:
                diff_text = ""

    # Persist the unified diff (may be empty)
    Path("local-sample-pr.diff").write_text(diff_text or "", encoding="utf-8")

    # File path list between base and head: try to use git name-only, fall back to parsing diff
    files_text = ""
    try:
        files_text = detector.repo.git.diff("--name-only", f"{base}..{head}")
    except Exception:
        try:
            files_text = detector.repo.git.diff(
                "--name-only", f"origin/{base}..origin/{head}"
            )
        except Exception:
            files_text = ""

    if not files_text and diff_text:
        # derive file paths from diff_text
        paths = []
        for line in diff_text.splitlines():
            if line.startswith("+++ "):
                parts = line.split()
                if len(parts) >= 2:
                    p = parts[1]
                    p = re.sub(r"^([ab]/)", "", p)
                    if p not in paths:
                        paths.append(p)
        files_text = "\n".join(paths)

    Path("local-sample-pr.files").write_text(files_text or "", encoding="utf-8")


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
        "--pr-url",
        help="PR URL to initialize the workflow with (optional)",
    )
    parser.add_argument(
        "--repo-path", default=".", help="Path to git repository (default: current dir)"
    )
    args = parser.parse_args(argv)

    provider = os.getenv("LLM_PROVIDER", "mock")
    logger.info("PR Review Assistant starting (LLM_PROVIDER={})", provider)

    try:
        # Generate input files if requested
        if args.commit:
            logger.info("Generating diff/files from commit: {}", args.commit)
            _generate_diff_and_files_from_commit(args.repo_path, args.commit)
        elif args.range:
            base, head = args.range
            logger.info("Generating diff/files from range: %s..%s", base, head)
            _generate_diff_and_files_from_range(args.repo_path, base, head)

        # Import here to keep startup lightweight when main isn't used
        from agents.planner import MVPOrchestrator

        async def _run() -> None:
            orch = MVPOrchestrator()
            pr_url = args.pr_url or "local-sample-pr"
            state = await orch.initialize_workflow(pr_url)
            state = await orch.run_mvp_pipeline(state)

            # Print and persist final report
            final = state.outputs.final_report or "{}"
            logger.info(final)
            Path("final_report.json").write_text(final, encoding="utf-8")
            # Persist posting metadata if available (from orchestrator)
            post_meta = getattr(state.outputs, "final_report_post", None)
            if post_meta:
                logger.info("FINAL_REPORT_POST:")
                logger.info(str(post_meta))
                try:
                    # Try to interpret as JSON and save as .json
                    parsed = json.loads(post_meta)
                    Path("final_report_post.json").write_text(
                        json.dumps(parsed, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                except Exception:
                    # Fallback to plain text
                    Path("final_report_post.txt").write_text(
                        str(post_meta), encoding="utf-8"
                    )

        import asyncio

        asyncio.run(_run())
    except Exception as e:
        logger.exception("Failed to run MVP orchestrator: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
