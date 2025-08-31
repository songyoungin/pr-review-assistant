"""
Git Change Detector Tool

This tool provides comprehensive functionality for detecting and analyzing changes
in Git repositories. It can:
- Detect changed files between commits or branches
- Extract detailed diff information
- Analyze commit metadata and history
- Identify the scope and impact of changes
"""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from git import Diff, Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from loguru import logger

from tools.base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult


@dataclass
class FileChange:
    """Represents a change in a single file."""

    file_path: str
    change_type: str  # 'A' (added), 'D' (deleted), 'M' (modified), 'R' (renamed)
    additions: int
    deletions: int
    diff_content: str
    similarity: float | None = None  # For renamed files


@dataclass
class CommitInfo:
    """Represents metadata about a commit."""

    commit_hash: str
    author: str
    author_email: str
    commit_date: str
    commit_message: str
    files_changed: list[str]


@dataclass
class GitAnalysisInput:
    """Input data for Git analysis."""

    repo_path: str
    base_ref: str | None = None
    head_ref: str | None = None
    commit_hash: str | None = None
    max_commits: int = 10


@dataclass
class GitAnalysisOutput:
    """Git analysis results."""

    changes: list[FileChange] | None = None
    commit_info: CommitInfo | None = None
    repository_stats: dict[str, Any] | None = None
    diff_summary: dict[str, Any] | None = None
    branch_summary: dict[str, Any] | None = None


class GitChangeDetector(BaseTool[GitAnalysisInput, GitAnalysisOutput]):
    """
    Git 변경사항을 감지하고 분석하는 포괄적인 툴.

    이 클래스는 다음 메서드들을 제공합니다:
    1. Git 저장소에 대한 연결 초기화
    2. 서로 다른 참조(커밋, 브랜치) 간의 변경사항 감지
    3. 변경사항의 범위와 영향 분석
    4. 리뷰를 위한 상세한 diff 정보 추출
    """

    def __init__(self, repo_path: str | None = None):
        """
        Git Change Detector 초기화.

        Args:
            repo_path: Git 저장소 경로 (상대 또는 절대 경로)
        """
        super().__init__("GitChangeDetector")
        self.repo_path: Path | None = None
        self.repo: Repo | None = None

        if repo_path:
            self._initialize_repository(repo_path)

    def execute(self, input_data: GitAnalysisInput) -> ToolResult[GitAnalysisOutput]:
        """
        Git 분석 실행.

        Args:
            input_data: Git 분석을 위한 입력 데이터

        Returns:
            Git 분석 결과
        """
        try:
            # 저장소 초기화
            if not self.repo:
                self._initialize_repository(input_data.repo_path)

            # 입력 검증
            if not self.validate_input(input_data):
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="잘못된 입력 데이터입니다",
                    metrics=self._create_metrics(),
                )

            # 분석 실행
            output = self._perform_analysis(input_data)

            # 증거 생성
            evidence = self._create_evidence(input_data, output)

            # 메트릭 생성
            metrics = self._create_metrics(
                files_processed=len(output.changes) if output.changes else 0
            )

            return ToolResult.success(
                output=output,
                evidence=evidence,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Git 분석 실행 중 오류 발생: {e}")
            return ToolResult.error(
                error_code=ToolErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                metrics=self._create_metrics(),
            )

    def _initialize_repository(self, repo_path: str) -> None:
        """
        Initialize Git repository.

        Args:
            repo_path: Git repository path

        Raises:
            InvalidGitRepositoryError: Repository path is not a valid Git repository
            ValueError: Repository path does not exist or is not a directory
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
            logger.info(f"Git repository initialized: {self.repo_path}")
        except InvalidGitRepositoryError as err:
            raise InvalidGitRepositoryError(
                f"Invalid Git repository: {self.repo_path}"
            ) from err

    def _perform_analysis(self, input_data: GitAnalysisInput) -> GitAnalysisOutput:
        """Perform Git analysis."""
        output = GitAnalysisOutput()

        # Get basic repository statistics
        output.repository_stats = self._get_repository_stats()

        # Analyze changes
        if input_data.base_ref and input_data.head_ref:
            output.changes = self.get_changed_files(
                input_data.base_ref, input_data.head_ref
            )
            output.diff_summary = self._get_diff_summary(
                input_data.base_ref, input_data.head_ref
            )
            output.branch_summary = self._get_branch_summary(
                input_data.base_ref, input_data.head_ref
            )

        # Analyze commit information
        if input_data.commit_hash:
            output.commit_info = self.get_commit_info(input_data.commit_hash)

        return output

    def _create_evidence(
        self, input_data: GitAnalysisInput, output: GitAnalysisOutput
    ) -> list[ToolEvidence]:
        """Create evidence for analysis results."""
        evidence = []

        # Repository information evidence
        if self.repo_path:
            evidence.append(
                ToolEvidence(
                    file_path=str(self.repo_path),
                    content=f"Git repository: {self.repo_path}",
                    evidence_type="repository",
                    description="Analyzed Git repository path",
                )
            )

        # Changes evidence
        if output.changes:
            evidence.append(
                ToolEvidence(
                    file_path="changes",
                    content=f"Total {len(output.changes)} files changed",
                    evidence_type="changes",
                    description="Detected file change summary",
                )
            )

        return evidence

    def _get_error_code(self, error_type: str) -> str:
        """Return error code based on error type."""
        error_mapping = {
            "INVALID_INPUT": "INVALID_INPUT",
            "PROCESSING_ERROR": "PROCESSING_ERROR",
            "FILE_NOT_FOUND": "FILE_NOT_FOUND",
        }
        return error_mapping.get(error_type, "PROCESSING_ERROR")

    def validate_input(self, input_data: GitAnalysisInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            Validation pass/fail
        """
        if not input_data.repo_path:
            return False

        if not Path(input_data.repo_path).exists():
            return False

        return True

    def get_changed_files(self, base_ref: str, head_ref: str) -> list[FileChange]:
        """
        Get all changed files between two Git references.

        Args:
            base_ref: Base reference (e.g., 'main', 'develop', or commit hash)
            head_ref: Head reference (e.g., feature branch or commit hash)

        Returns:
            List of FileChange objects representing all changes

        Raises:
            GitCommandError: Git command failed
            ValueError: Reference is invalid
        """
        if not self.repo:
            raise ValueError("Git repository not initialized")

        try:
            # Get diff between two references
            base_commit = self.repo.commit(base_ref)
            head_commit = self.repo.commit(head_ref)

            # Create diff
            diff = base_commit.diff(head_commit)

            # Process each change
            changes = []
            for change in diff:
                file_change = self._process_diff_change(change)
                if file_change:
                    changes.append(file_change)

            logger.info(
                f"{base_ref} and {head_ref} have {len(changes)} file changes detected"
            )
            return changes

        except GitCommandError as e:
            logger.error(f"Git command failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting changed files: {e}")
            raise

    def _process_diff_change(self, diff: Diff) -> FileChange | None:
        """
        Process a single diff change and convert it to a FileChange object.

        Args:
            diff: Git diff object

        Returns:
            FileChange object or None if unable to process
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
                logger.warning("No valid file path in diff change, skipping")
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
        Get detailed information for a specific commit.

        Args:
            commit_ref: Commit reference (hash, branch name, etc.)

        Returns:
            CommitInfo object with commit metadata

        Raises:
            GitCommandError: Git command failed
        """
        if not self.repo:
            raise ValueError("Git repository not initialized")

        try:
            commit = self.repo.commit(commit_ref)

            # Get changed files in this commit
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

    def _get_repository_stats(self) -> dict[str, Any]:
        """Get general statistics for the repository."""
        if not self.repo:
            return {}

        try:
            # Active branch
            active_branch = self.repo.active_branch.name

            # Total commits
            total_commits = len(list(self.repo.iter_commits()))

            # Recent commits (last 30 days)
            cutoff_date = datetime.now(UTC) - timedelta(days=30)
            recent_commits = len(
                [
                    commit
                    for commit in self.repo.iter_commits()
                    if commit.authored_datetime.replace(tzinfo=UTC) > cutoff_date
                ]
            )

            # File count
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
            logger.error(f"Failed to get repository statistics: {e}")
            return {}

    def _get_diff_summary(self, base_ref: str, head_ref: str) -> dict[str, Any]:
        """Get summary of changes between two references."""
        try:
            changes = self.get_changed_files(base_ref, head_ref)

            # Calculate statistics
            total_additions = sum(change.additions for change in changes)
            total_deletions = sum(change.deletions for change in changes)

            # Group by change type
            change_types: dict[str, list[str]] = {}
            for change in changes:
                change_type = change.change_type
                if change_type not in change_types:
                    change_types[change_type] = []
                change_types[change_type].append(change.file_path)

            # File extensions
            file_extensions: dict[str, int] = {}
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

    def _get_branch_summary(self, base_branch: str, head_branch: str) -> dict[str, Any]:
        """Get comprehensive summary of changes between two branches."""
        if not self.repo:
            return {}

        try:
            # Get basic diff summary
            diff_summary = self._get_diff_summary(base_branch, head_branch)

            # Get commit information for head branch
            head_commit = self.repo.heads[head_branch].commit
            base_commit = self.repo.heads[base_branch].commit

            # Calculate number of commits between branches
            commit_count = 0
            for _commit in self.repo.iter_commits(f"{base_branch}..{head_branch}"):
                commit_count += 1

            # Author statistics
            authors: dict[str, int] = {}
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

            logger.info(f"{base_branch} vs {head_branch} comprehensive summary created")
            return enhanced_summary

        except Exception as e:
            logger.error(f"Failed to get branch summary: {e}")
            return {}
