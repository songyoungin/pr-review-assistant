"""Source Code Management Tools.
This tool provides a set of tools for managing source code.

1. PR metadata retrieval: base/head SHA, branches, author
2. Changed files list + file-level diff (hunks) retrieval
3. File content snapshots: both base and head timepoints
4. (Optional) blame metadata (line-by-line author/commit)
5. Rate limiting/retry/pagination/cache handling
"""

from __future__ import annotations

import os
import time
import requests

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Any
from loguru import logger


@dataclass(frozen=True)
class PullRequestInfo:
    """Information about a pull request."""

    number: int
    base_sha: str
    head_sha: str
    base_ref: str
    head_ref: str


@dataclass(frozen=True)
class FileChange:
    """Information about a file change."""

    path: str  # path to modified file
    status: str  # added/modified/removed/renamed
    additions: int
    deletions: int
    changes: int
    previous_filename: Optional[str] = None
    patch: Optional[str] = None  # unified diff (optional)


@dataclass(frozen=True)
class Hunk:
    """A single hunk of a file change."""

    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    lines: List[str]


class SCM(Protocol):
    """Protocol for interacting with a source code management system."""

    def get_pr_info(self, repo: str, pr_number: int) -> PullRequestInfo: ...
    def list_changed_files(
        self, repo: str, pr_number: int, *, with_patch: bool = True
    ) -> List[FileChange]: ...
    def iter_hunks(self, patch: str) -> Iterable[Hunk]: ...
    def get_file_at(self, repo: str, path: str, ref: str) -> Optional[str]: ...
    def is_binary(self, repo: str, path: str, ref: str) -> bool: ...

    # Optional
    def get_blame(self, repo: str, path: str, ref: str) -> Optional[dict]: ...


class GitHubSCM(SCM):
    """GitHub implementation of the SCM protocol."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = "https://api.github.com",
        timeout: int = 30,
    ):
        """Initialize the GitHubSCM.

        Args:
            token (Optional[str], optional): GitHub API token. Defaults to None.
            base_url (str, optional): GitHub API base URL. Defaults to "https://api.github.com".
            timeout (int, optional): Request timeout in seconds. Defaults to 30.
        """
        self.token = token or os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        self.session.headers.update({"Accept": "application/vnd.github+json"})
        self.timeout = timeout

    # --- Utility: pagination/retry/rate limit ---
    def _get(self, url: str, params: Optional[dict] = None) -> requests.Response:
        """Make a GET request with rate limit handling.

        Args:
            url (str): The URL to make the request to.
            params (Optional[dict], optional): The parameters to pass to the request. Defaults to None.

        Returns:
            requests.Response: The response from the request.
        """
        backoff = 1.0
        for _ in range(5):
            r = self.session.get(url, params=params, timeout=self.timeout)

            # Handle rate limit
            if (
                r.status_code == 403
                and "X-RateLimit-Remaining" in r.headers
                and r.headers.get("X-RateLimit-Remaining") == "0"
            ):
                reset = int(r.headers.get("X-RateLimit-Reset", "1"))
                sleep_for = max(1, reset - int(time.time()) + 1)
                time.sleep(sleep_for)
                continue

            # Retry on temporary errors
            if r.status_code >= 500:
                time.sleep(backoff)
                backoff *= 2
                continue

            r.raise_for_status()
            return r

        r.raise_for_status()
        return r  # pragma: no cover

    def _paginate(self, url: str, params: Optional[dict] = None) -> Iterable[Any]:
        """Paginate through a list of items.

        Args:
            url (str): The URL to make the request to.
            params (Optional[dict], optional): The parameters to pass to the request. Defaults to None.

        Returns:
            Iterable[Any]: An iterator over the items.
        """
        page = 1

        while True:
            p = dict(params or {})
            p.update({"page": page, "per_page": 100})
            r = self._get(url, p)
            items = r.json()

            # Break if no items
            if not items:
                break
            yield from items

            # Break if less than 100 items
            if len(items) < 100:
                break

            page += 1

    # --- Interface implementation ---
    def get_pr_info(self, repo: str, pr_number: int) -> PullRequestInfo:
        """Get information about a pull request.

        Args:
            repo (str): The repository name.
            pr_number (int): The pull request number.

        Returns:
            PullRequestInfo: Information about the pull request.
        """
        url = f"{self.base_url}/repos/{repo}/pulls/{pr_number}"
        data = self._get(url).json()

        return PullRequestInfo(
            number=data["number"],
            base_sha=data["base"]["sha"],
            head_sha=data["head"]["sha"],
            base_ref=data["base"]["ref"],
            head_ref=data["head"]["ref"],
        )

    def list_changed_files(
        self, repo: str, pr_number: int, *, with_patch: bool = True
    ) -> List[FileChange]:
        """List the files changed in a pull request.

        Args:
            repo (str): The repository name.
            pr_number (int): The pull request number.
            with_patch (bool, optional): Whether to include the patch in the response. Defaults to True.
        """
        url = f"{self.base_url}/repos/{repo}/pulls/{pr_number}/files"
        files: List[FileChange] = []

        for item in self._paginate(url):
            if not with_patch:
                item.pop("patch", None)

            files.append(
                FileChange(
                    path=item["filename"],
                    status=item["status"],
                    additions=item["additions"],
                    deletions=item["deletions"],
                    changes=item["changes"],
                    previous_filename=item.get("previous_filename"),
                    patch=item.get("patch"),
                )
            )

        return files

    def iter_hunks(self, patch: str) -> Iterable[Hunk]:
        """Iterate over the hunks in a patch.

        Args:
            patch (str): The patch to iterate over.

        Returns:
            Iterable[Hunk]: An iterator over the hunks in the patch.
        """
        raise NotImplementedError("Not implemented yet.")

    def get_file_at(self, repo: str, path: str, ref: str) -> Optional[str]:
        """Get the content of a file at a given reference.

        Args:
            repo (str): The repository name.
            path (str): The path to the file.
            ref (str): The reference to get the file from.
        """
        raise NotImplementedError("Not implemented yet.")

    def is_binary(self, repo: str, path: str, ref: str) -> bool:
        """Check if a file is binary.

        Args:
            repo (str): The repository name.
            path (str): The path to the file.
            ref (str): The reference to get the file from.
        """
        raise NotImplementedError("Not implemented yet.")

    def get_blame(self, repo: str, path: str, ref: str) -> Optional[dict]:
        """Get the blame information for a file at a given reference.

        Args:
            repo (str): The repository name.
            path (str): The path to the file.
            ref (str): The reference to get the blame from.
        """
        raise NotImplementedError("Not implemented yet.")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    scm = GitHubSCM(
        token=os.getenv("GITHUB_TOKEN"),
    )
    pr_info = scm.get_pr_info("langchain-ai/langgraph", 1)
    logger.info(pr_info)

    changed_files = scm.list_changed_files("langchain-ai/langgraph", 1)
    logger.info(changed_files)
