"""
GitHub provider utility for posting PR comments.

This module provides a small, dependency-light wrapper around the GitHub
issues/comments API to post a comment to a pull request. It implements
exponential backoff retry for transient failures and a URL parser to
extract owner/repo/number from common PR URL formats.

The implementation intentionally uses the `requests` library for clarity.
"""

from __future__ import annotations

import re
import time
from typing import Any

import requests
from loguru import logger


def parse_github_pr_url(pr_url: str) -> tuple[str, str, int]:
    """Parse a GitHub PR URL and return (owner, repo, number).

    Supported forms include:
      - https://github.com/{owner}/{repo}/pull/{number}
      - git@github.com:{owner}/{repo}.git with separate PR number not encoded

    Raises ValueError if parsing fails.
    """
    if not pr_url:
        raise ValueError("Empty PR URL")

    # Common web URL form
    m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
    if m:
        owner, repo, num = m.group(1), m.group(2), int(m.group(3))
        return owner, repo, num

    # SSH short form: git@github.com:owner/repo.git
    m2 = re.search(r"git@github\.com:([^/]+)/([^.]+)(?:\.git)?", pr_url)
    if m2:
        owner, repo = m2.group(1), m2.group(2)
        raise ValueError(
            "PR number not found in SSH repo URL; supply PR number separately"
        )

    raise ValueError(f"Unable to parse GitHub PR URL: {pr_url}")


class GitHubPoster:
    """Simple poster for GitHub PR comments.

    Usage:
        poster = GitHubPoster(token=os.getenv("GITHUB_TOKEN"))
        poster.post_comment("owner", "repo", 123, "hello")
    """

    def __init__(self, token: str, api_url: str = "https://api.github.com") -> None:
        if not token:
            raise ValueError("GitHub token is required")
        self.token = token
        self.api_url = api_url.rstrip("/")

    def post_comment(
        self, owner: str, repo: str, pr_number: int, body: str, timeout: int = 10
    ) -> dict[str, Any]:
        """Post a comment to the given PR and return the parsed JSON response.

        Retries transient errors with exponential backoff (3 attempts).
        Raises RuntimeError on persistent failure.
        """
        url = f"{self.api_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "pr-review-assistant",
        }
        payload = {"body": body}

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(f"Posting PR comment attempt {attempt} to {url}")
                resp = requests.post(
                    url, headers=headers, json=payload, timeout=timeout
                )
                if resp.ok:
                    try:
                        data = resp.json()
                    except Exception:
                        return {"raw_text": resp.text}

                    # Ensure we return a dict[str, Any] to satisfy static type
                    # checkers; fall back to stringified form when necessary.
                    if isinstance(data, dict):
                        return data
                    return {"raw_text": str(data)}

                # Authentication/authorization errors should not be retried
                if resp.status_code in (401, 403):
                    logger.error(
                        "Auth error posting comment: %s %s", resp.status_code, resp.text
                    )
                    raise RuntimeError(
                        f"Auth error posting comment: {resp.status_code}"
                    )

                logger.warning(
                    "Non-ok response posting comment (attempt %s): %s %s",
                    attempt,
                    resp.status_code,
                    resp.text,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "RequestException posting comment (attempt %s): %s", attempt, exc
                )

            # Backoff before next attempt if any
            if attempt < max_attempts:
                sleep_sec = 2 ** (attempt - 1)
                time.sleep(sleep_sec)

        raise RuntimeError("Failed to post comment after retries")
