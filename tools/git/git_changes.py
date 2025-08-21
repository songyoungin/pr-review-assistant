"""
Git Change Detector Tool

This tool provides comprehensive functionality for detecting and analyzing changes
in Git repositories. It can:
- Detect changed files between commits or branches
- Extract detailed diff information
- Analyze commit metadata and history
- Identify the scope and impact of changes
"""

import sys

from pathlib import Path
from datetime import datetime, timedelta, timezone

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from git import Repo, Diff
from git.exc import GitCommandError, InvalidGitRepositoryError
from loguru import logger


@dataclass
class FileChange:
    """Represents a change in a single file."""

    file_path: str
    change_type: str  # 'A' (added), 'D' (deleted), 'M' (modified), 'R' (renamed)
    additions: int
    deletions: int
    diff_content: str
    similarity: Optional[float] = None  # For renamed files


@dataclass
class CommitInfo:
    """Represents metadata about a commit."""

    commit_hash: str
    author: str
    author_email: str
    commit_date: str
    commit_message: str
    files_changed: List[str]


class GitChangeDetector:
    """
    A comprehensive tool for detecting and analyzing Git changes.

    This class provides methods to:
    1. Initialize connection to a Git repository
    2. Detect changes between different references (commits, branches)
    3. Analyze the scope and impact of changes
    4. Extract detailed diff information for review
    """

    def __init__(self, repo_path: str):
        """
        Initialize the Git Change Detector.

        Args:
            repo_path: Path to the Git repository (can be relative or absolute)

        Raises:
            InvalidGitRepositoryError: If the path is not a valid Git repository
            ValueError: If the repository path doesn't exist
        """
        self.repo_path = Path(repo_path).resolve()

        # Validate repository path
        if not self.repo_path.exists():
            raise ValueError(f"Repository path does not exist: {self.repo_path}")

        if not self.repo_path.is_dir():
            raise ValueError(f"Repository path is not a directory: {self.repo_path}")

        # Initialize Git repository
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Successfully initialized Git repository at {self.repo_path}")
        except InvalidGitRepositoryError:
            raise InvalidGitRepositoryError(
                f"Invalid Git repository at {self.repo_path}"
            )

    def get_changed_files(self, base_ref: str, head_ref: str) -> List[FileChange]:
        """
        Get all changed files between two Git references.

        Args:
            base_ref: Base reference (e.g., 'main', 'develop', or commit hash)
            head_ref: Head reference (e.g., feature branch or commit hash)

        Returns:
            List of FileChange objects representing all changes

        Raises:
            GitCommandError: If Git command fails
            ValueError: If references are invalid
        """
        try:
            # Get the diff between the two references
            base_commit = self.repo.commit(base_ref)
            head_commit = self.repo.commit(head_ref)

            # Generate diff
            diff = base_commit.diff(head_commit)

            # Process each change
            changes = []
            for change in diff:
                file_change = self._process_diff_change(change)
                if file_change:
                    changes.append(file_change)

            logger.info(
                f"Detected {len(changes)} file changes between {base_ref} and {head_ref}"
            )
            return changes

        except GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while getting changed files: {e}")
            raise

    def _process_diff_change(self, diff: Diff) -> Optional[FileChange]:
        """
        Process a single diff change and convert it to a FileChange object.

        Args:
            diff: Git diff object

        Returns:
            FileChange object or None if the change cannot be processed
        """
        try:
            # Determine change type and file path
            if diff.new_file:
                change_type = "A"  # Added
                file_path = diff.b_path or ""
            elif diff.deleted_file:
                change_type = "D"  # Deleted
                file_path = diff.a_path or ""
            elif diff.renamed_file:
                change_type = "R"  # Renamed
                file_path = diff.b_path or ""
            else:
                change_type = "M"  # Modified
                file_path = diff.b_path or ""

            # Skip if no valid file path
            if not file_path:
                logger.warning("Skipping diff change with no valid file path")
                return None

            # Get diff statistics safely
            try:
                additions = (
                    diff.stats.get("insertions", 0) if hasattr(diff, "stats") else 0
                )
                deletions = (
                    diff.stats.get("deletions", 0) if hasattr(diff, "stats") else 0
                )
            except AttributeError:
                additions = 0
                deletions = 0

            # Get diff content safely
            try:
                diff_content = (
                    diff.diff.decode("utf-8", errors="ignore") if diff.diff else ""
                )
            except (AttributeError, UnicodeDecodeError):
                diff_content = ""

            # Calculate similarity for renamed files
            similarity = None
            if change_type == "R":
                similarity = getattr(diff, "rename_score", None)

            return FileChange(
                file_path=file_path,
                change_type=change_type,
                additions=additions,
                deletions=deletions,
                diff_content=diff_content,
                similarity=similarity,
            )

        except Exception as e:
            logger.warning(f"Failed to process diff change: {e}")
            return None

    def get_commit_info(self, commit_ref: str) -> CommitInfo:
        """
        Get detailed information about a specific commit.

        Args:
            commit_ref: Commit reference (hash, branch name, etc.)

        Returns:
            CommitInfo object with commit metadata

        Raises:
            GitCommandError: If Git command fails
        """
        try:
            commit = self.repo.commit(commit_ref)

            # Get files changed in this commit
            files_changed = []
            if commit.parents:
                # Compare with parent commit
                diff = commit.parents[0].diff(commit)
                files_changed = [change.b_path for change in diff if change.b_path]
            else:
                # Root commit - all files are new
                files_changed = []
                for item in commit.tree.traverse():
                    if (
                        hasattr(item, "type")
                        and item.type == "blob"
                        and hasattr(item, "a_path")
                    ):
                        files_changed.append(item.a_path)

            return CommitInfo(
                commit_hash=commit.hexsha,
                author=commit.author.name or "Unknown",
                author_email=commit.author.email or "Unknown",
                commit_date=commit.authored_datetime.isoformat(),
                commit_message=(
                    commit.message.strip()
                    if isinstance(commit.message, str)
                    else commit.message.decode("utf-8", errors="ignore").strip()
                ),
                files_changed=files_changed,
            )

        except GitCommandError as e:
            logger.error(f"Failed to get commit info for {commit_ref}: {e}")
            raise

    def get_file_history(
        self, file_path: str, max_commits: int = 10
    ) -> List[CommitInfo]:
        """
        Get the commit history for a specific file.

        Args:
            file_path: Path to the file relative to repository root
            max_commits: Maximum number of commits to return

        Returns:
            List of CommitInfo objects representing the file's history

        Raises:
            GitCommandError: If Git command fails
            ValueError: If file path is invalid
        """
        try:
            # Validate file path
            full_path = self.repo_path / file_path
            if not full_path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            # Get commit history for the file
            commits = list(
                self.repo.iter_commits(paths=file_path, max_count=max_commits)
            )

            # Convert to CommitInfo objects
            history = []
            for commit in commits:
                commit_info = self.get_commit_info(commit.hexsha)
                history.append(commit_info)

            logger.info(f"Retrieved {len(history)} commits for file {file_path}")
            return history

        except GitCommandError as e:
            logger.error(f"Failed to get file history for {file_path}: {e}")
            raise

    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get general statistics about the repository.

        Returns:
            Dictionary containing repository statistics
        """
        try:
            # Get active branch
            active_branch = self.repo.active_branch.name

            # Get total commits
            total_commits = len(list(self.repo.iter_commits()))

            # Get recent commits (last 30 days)

            cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
            recent_commits = len(
                [
                    commit
                    for commit in self.repo.iter_commits()
                    if commit.authored_datetime.replace(tzinfo=timezone.utc)
                    > cutoff_date
                ]
            )

            # Get file count
            file_count = len(
                [
                    item
                    for item in self.repo.head.commit.tree.traverse()
                    if item.type == "blob"
                ]
            )

            return {
                "active_branch": active_branch,
                "total_commits": total_commits,
                "recent_commits_30d": recent_commits,
                "total_files": file_count,
                "repository_path": str(self.repo_path),
                "is_dirty": self.repo.is_dirty(),
            }

        except Exception as e:
            logger.error(f"Failed to get repository stats: {e}")
            return {}

    def get_diff_summary(self, base_ref: str, head_ref: str) -> Dict[str, Any]:
        """
        Get a summary of changes between two references.

        Args:
            base_ref: Base reference
            head_ref: Head reference

        Returns:
            Dictionary containing change summary
        """
        try:
            changes = self.get_changed_files(base_ref, head_ref)

            # Calculate statistics
            total_additions = sum(change.additions for change in changes)
            total_deletions = sum(change.deletions for change in changes)

            # Group by change type
            change_types: Dict[str, List[str]] = {}
            for change in changes:
                change_type = change.change_type
                if change_type not in change_types:
                    change_types[change_type] = []
                change_types[change_type].append(change.file_path)

            # Get file extensions
            file_extensions: Dict[str, int] = {}
            for change in changes:
                if change.file_path:
                    ext = Path(change.file_path).suffix
                    if ext not in file_extensions:
                        file_extensions[ext] = 0
                    file_extensions[ext] += 1

            return {
                "total_files_changed": len(changes),
                "total_additions": total_additions,
                "total_deletions": total_deletions,
                "net_change": total_additions - total_deletions,
                "change_types": change_types,
                "file_extensions": file_extensions,
                "base_ref": base_ref,
                "head_ref": head_ref,
            }

        except Exception as e:
            logger.error(f"Failed to get diff summary: {e}")
            return {}

    def get_commit_changes(self, commit_hash: str) -> List[FileChange]:
        """
        Get changes introduced by a specific commit.

        Args:
            commit_hash: Hash of the commit to analyze

        Returns:
            List of FileChange objects representing changes in this commit

        Raises:
            GitCommandError: If Git command fails
            ValueError: If commit hash is invalid
        """
        try:
            commit = self.repo.commit(commit_hash)

            if not commit.parents:
                # Root commit - all files are new
                logger.info(f"Root commit {commit_hash[:8]} - all files are new")
                return []

            # Compare with parent commit
            parent_commit = commit.parents[0]
            changes = self.get_changed_files(parent_commit.hexsha, commit_hash)

            logger.info(f"Found {len(changes)} changes in commit {commit_hash[:8]}")
            return changes

        except GitCommandError as e:
            logger.error(f"Git command failed for commit {commit_hash}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to get commit changes for {commit_hash}: {e}")
            raise

    def get_range_changes(self, start_commit: str, end_commit: str) -> List[FileChange]:
        """
        Get changes between two specific commits (range analysis).

        Args:
            start_commit: Starting commit hash
            end_commit: Ending commit hash

        Returns:
            List of FileChange objects representing changes in the range

        Raises:
            GitCommandError: If Git command fails
            ValueError: If commit hashes are invalid
        """
        try:
            # Get changes between the two commits
            changes = self.get_changed_files(start_commit, end_commit)

            logger.info(
                f"Found {len(changes)} changes between commits {start_commit[:8]} and {end_commit[:8]}"
            )
            return changes

        except GitCommandError as e:
            logger.error(
                f"Git command failed for range {start_commit[:8]}..{end_commit[:8]}: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to get range changes: {e}")
            raise

    def get_branch_changes(
        self, base_branch: str, head_branch: str
    ) -> List[FileChange]:
        """
        Get all changes between two branches (alias for get_changed_files).

        Args:
            base_branch: Base branch name
            head_branch: Head branch name

        Returns:
            List of FileChange objects representing all changes between branches

        Note:
            This is an alias for get_changed_files() for better semantic clarity
            when working with branches specifically.
        """
        return self.get_changed_files(base_branch, head_branch)

    def get_commit_info_detailed(self, commit_hash: str) -> Dict[str, Any]:
        """
        Get detailed information about a commit including its changes.

        Args:
            commit_hash: Hash of the commit to analyze

        Returns:
            Dictionary containing detailed commit information and changes
        """
        try:
            commit_info = self.get_commit_info(commit_hash)

            # Get changes in this commit
            changes = self.get_commit_changes(commit_hash)

            # Calculate change statistics
            total_additions = sum(change.additions for change in changes)
            total_deletions = sum(change.deletions for change in changes)

            # Group files by change type
            change_types: Dict[str, List[str]] = {}
            for change in changes:
                change_type = change.change_type
                if change_type not in change_types:
                    change_types[change_type] = []
                change_types[change_type].append(change.file_path)

            return {
                "commit_info": commit_info,
                "changes": changes,
                "statistics": {
                    "total_files_changed": len(changes),
                    "total_additions": total_additions,
                    "total_deletions": total_deletions,
                    "net_change": total_additions - total_deletions,
                    "change_types": change_types,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get detailed commit info for {commit_hash}: {e}")
            return {}

    def get_branch_summary(self, base_branch: str, head_branch: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of changes between two branches.

        Args:
            base_branch: Base branch name
            head_branch: Head branch name

        Returns:
            Dictionary containing comprehensive branch comparison summary
        """
        try:
            # Get basic diff summary
            diff_summary = self.get_diff_summary(base_branch, head_branch)

            # Get commit information for the head branch
            head_commit = self.repo.heads[head_branch].commit
            base_commit = self.repo.heads[base_branch].commit

            # Count commits between branches
            commit_count = 0
            for commit in self.repo.iter_commits(f"{base_branch}..{head_branch}"):
                commit_count += 1

            # Get author statistics
            authors: Dict[str, int] = {}
            for commit in self.repo.iter_commits(f"{base_branch}..{head_branch}"):
                author = commit.author.name or "Unknown"
                authors[author] = authors.get(author, 0) + 1

            # Enhanced summary
            enhanced_summary = {
                **diff_summary,
                "commit_count": commit_count,
                "authors": authors,
                "base_branch": base_branch,
                "head_branch": head_branch,
                "base_commit": base_commit.hexsha[:8],
                "head_commit": head_commit.hexsha[:8],
            }

            logger.info(
                f"Generated comprehensive summary for {base_branch} vs {head_branch}"
            )
            return enhanced_summary

        except Exception as e:
            logger.error(f"Failed to get branch summary: {e}")
            return {}


def main():
    """
    Example usage of GitChangeDetector class.

    This demonstrates how to use the GitChangeDetector to:
    1. Initialize connection to a Git repository
    2. Get changed files between references (branches or commits)
    3. Analyze commit information and changes
    4. Generate comprehensive change summaries
    5. Use commit-specific and range analysis methods

    Usage Examples:
        # Run in current directory (must be a Git repository)
        python git_changes.py

        # Run in a specific Git repository
        python git_changes.py /path/to/your/repo

        # Run from project root
        python tools/git/git_changes.py

        # Run with absolute path
        python tools/git/git_changes.py /Users/username/projects/my-repo

    PR-like Branch Comparison:
        The script automatically detects your current branch and compares it with
        the main branch (main/master/develop) to show changes like in a PR:

        # If you're on feature-branch, it will compare with main:
        git checkout feature-branch
        python git_changes.py

        # This will show all changes between feature-branch and main
        # Similar to what you'd see in a Pull Request

    Command Line Arguments:
        repo_path (optional): Path to the Git repository
                             Default: current directory (.)

    Output:
        - Repository statistics and metadata
        - Detailed analysis of changes between HEAD and parent commit
        - File-by-file change information with diff previews
        - Change summary with statistics and breakdowns
        - Current HEAD commit information
        - Sample file history (if Python files exist)

    Requirements:
        - Must be run from within a Git repository
        - Repository must have at least one commit
        - Python packages: gitpython, loguru
    """

    # Check if repository path is provided as command line argument
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        # Default to current directory
        repo_path = "."

    try:
        # Initialize Git Change Detector
        logger.info(f"Initializing Git Change Detector for repository: {repo_path}")
        detector = GitChangeDetector(repo_path)

        # Get repository statistics
        logger.info("Getting repository statistics...")
        stats = detector.get_repository_stats()
        logger.info(f"Repository stats: {stats}")

        # Example: Get changes between current branch and main branch (PR-like scenario)
        try:
            # Get current branch name
            current_branch = detector.repo.active_branch.name
            logger.info(f"Current branch: {current_branch}")

            # Check if main branch exists (try common names)
            main_branch_names = ["main", "master", "develop"]
            main_branch = None

            for branch_name in main_branch_names:
                try:
                    detector.repo.heads[branch_name]
                    main_branch = branch_name
                    break
                except IndexError:
                    continue

            if main_branch and main_branch != current_branch:
                logger.info(
                    f"Comparing {current_branch} branch with {main_branch} branch"
                )

                # Get changes between current branch and main branch
                changes = detector.get_changed_files(main_branch, current_branch)
                logger.info(f"Found {len(changes)} changed files")

                # Display detailed changes
                for i, change in enumerate(changes, 1):
                    logger.info(f"\nChange {i}:")
                    logger.info(f"  File: {change.file_path}")
                    logger.info(f"  Type: {change.change_type}")
                    logger.info(f"  Additions: {change.additions}")
                    logger.info(f"  Deletions: {change.deletions}")

                    if change.change_type == "R" and change.similarity:
                        logger.info(f"  Rename similarity: {change.similarity}")

                    # Show first few lines of diff (truncated for readability)
                    diff_preview = (
                        change.diff_content[:200] + "..."
                        if len(change.diff_content) > 200
                        else change.diff_content
                    )
                    logger.info(f"  Diff preview:\n{diff_preview}")

                # Get change summary
                summary = detector.get_diff_summary(main_branch, current_branch)
                logger.info(f"\nChange Summary ({current_branch} vs {main_branch}):")
                logger.info(f"  Total files changed: {summary['total_files_changed']}")
                logger.info(f"  Total additions: {summary['total_additions']}")
                logger.info(f"  Total deletions: {summary['total_deletions']}")
                logger.info(f"  Net change: {summary['net_change']}")

                # Show change types breakdown
                logger.info("  Change types:")
                for change_type, files in summary["change_types"].items():
                    logger.info(f"    {change_type}: {len(files)} files")

                # Show file extensions breakdown
                logger.info("  File extensions:")
                for ext, count in summary["file_extensions"].items():
                    logger.info(f"    {ext or 'no extension'}: {count} files")

            elif main_branch == current_branch:
                logger.info(f"Currently on {main_branch} branch - no comparison needed")

                # Fallback: compare with parent commit
                head_commit = detector.repo.head.commit
                if head_commit.parents:
                    base_ref = head_commit.parents[0].hexsha
                    head_ref = head_commit.hexsha

                    logger.info(
                        f"Falling back to parent commit comparison: {base_ref[:8]} vs {head_ref[:8]}"
                    )
                    changes = detector.get_changed_files(base_ref, head_ref)
                    logger.info(f"Found {len(changes)} changed files in last commit")

            else:
                logger.warning(
                    f"Could not find main branch. Available branches: {[b.name for b in detector.repo.heads]}"
                )

        except Exception as e:
            logger.error(f"Error analyzing branch changes: {e}")

        # Example: Get commit information for current HEAD
        try:
            head_commit = detector.repo.head.commit
            commit_info = detector.get_commit_info(head_commit.hexsha)

            logger.info("\nCurrent HEAD commit information:")
            logger.info(f"  Hash: {commit_info.commit_hash}")
            logger.info(f"  Author: {commit_info.author} <{commit_info.author_email}>")
            logger.info(f"  Date: {commit_info.commit_date}")
            logger.info(f"  Message: {commit_info.commit_message}")
            logger.info(f"  Files changed: {len(commit_info.files_changed)}")

        except Exception as e:
            logger.error(f"Error getting commit info: {e}")

        # Example: Demonstrate new commit-specific methods
        try:
            head_commit = detector.repo.head.commit

            if head_commit.parents:
                # Get detailed commit information
                detailed_info = detector.get_commit_info_detailed(head_commit.hexsha)

                if detailed_info:
                    logger.info(
                        f"\nDetailed commit analysis for {head_commit.hexsha[:8]}:"
                    )
                    logger.info(f"  Statistics: {detailed_info['statistics']}")

                    # Show changes by type
                    for change_type, files in detailed_info["statistics"][
                        "change_types"
                    ].items():
                        logger.info(f"    {change_type}: {len(files)} files")

        except Exception as e:
            logger.error(f"Error getting detailed commit info: {e}")

        # Example: Demonstrate branch summary method
        try:
            current_branch = detector.repo.active_branch.name

            # Try to find main branch for comparison
            main_branch_names = ["main", "master", "develop"]
            main_branch = None

            for branch_name in main_branch_names:
                try:
                    detector.repo.heads[branch_name]
                    main_branch = branch_name
                    break
                except IndexError:
                    continue

            if main_branch and main_branch != current_branch:
                logger.info("\nGenerating comprehensive branch summary...")
                branch_summary = detector.get_branch_summary(
                    main_branch, current_branch
                )

                if branch_summary:
                    logger.info(f"Branch Summary ({current_branch} vs {main_branch}):")
                    logger.info(
                        f"  Total commits: {branch_summary.get('commit_count', 0)}"
                    )
                    logger.info(f"  Authors: {branch_summary.get('authors', {})}")
                    logger.info(
                        f"  Base commit: {branch_summary.get('base_commit', 'N/A')}"
                    )
                    logger.info(
                        f"  Head commit: {branch_summary.get('head_commit', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"Error generating branch summary: {e}")

        # Example: Get file history for a specific file (if it exists)
        try:
            # Look for Python files in the repository
            python_files = []
            for item in detector.repo.head.commit.tree.traverse():
                if (
                    hasattr(item, "type")
                    and item.type == "blob"
                    and hasattr(item, "a_path")
                ):
                    if item.a_path and item.a_path.endswith(".py"):
                        python_files.append(item.a_path)

            if python_files:
                # Get history for the first Python file found
                sample_file = python_files[0]
                logger.info(f"\nGetting file history for: {sample_file}")

                history = detector.get_file_history(sample_file, max_commits=3)
                logger.info(f"Found {len(history)} commits in history")

                for i, commit_info in enumerate(history, 1):
                    logger.info(
                        f"  Commit {i}: {commit_info.commit_hash[:8]} - {commit_info.commit_message[:50]}..."
                    )

        except Exception as e:
            logger.error(f"Error getting file history: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize Git Change Detector: {e}")
        logger.error(
            "Make sure you're in a valid Git repository or provide a valid path"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
