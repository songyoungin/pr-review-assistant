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
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from git import Diff, Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from loguru import logger

from ..base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult


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
        Git 저장소 초기화.

        Args:
            repo_path: Git 저장소 경로

        Raises:
            InvalidGitRepositoryError: 경로가 유효한 Git 저장소가 아닌 경우
            ValueError: 저장소 경로가 존재하지 않거나 디렉토리가 아닌 경우
        """
        self.repo_path = Path(repo_path).resolve()

        # 저장소 경로 검증
        if not self.repo_path.exists():
            raise ValueError(f"저장소 경로가 존재하지 않습니다: {self.repo_path}")

        if not self.repo_path.is_dir():
            raise ValueError(f"저장소 경로가 디렉토리가 아닙니다: {self.repo_path}")

        # Git 저장소 초기화
        try:
            self.repo = Repo(self.repo_path)
            logger.info(f"Git 저장소 초기화 성공: {self.repo_path}")
        except InvalidGitRepositoryError as err:
            raise InvalidGitRepositoryError(
                f"유효하지 않은 Git 저장소: {self.repo_path}"
            ) from err

    def _perform_analysis(self, input_data: GitAnalysisInput) -> GitAnalysisOutput:
        """Git 분석 수행."""
        output = GitAnalysisOutput()

        # 기본 저장소 통계
        output.repository_stats = self._get_repository_stats()

        # 변경사항 분석
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

        # 커밋 정보 분석
        if input_data.commit_hash:
            output.commit_info = self.get_commit_info(input_data.commit_hash)

        return output

    def _create_evidence(
        self, input_data: GitAnalysisInput, output: GitAnalysisOutput
    ) -> list[ToolEvidence]:
        """분석 결과에 대한 증거 생성."""
        evidence = []

        # 저장소 정보 증거
        if self.repo_path:
            evidence.append(
                ToolEvidence(
                    file_path=str(self.repo_path),
                    content=f"Git 저장소: {self.repo_path}",
                    evidence_type="repository",
                    description="분석된 Git 저장소 경로",
                )
            )

        # 변경사항 증거
        if output.changes:
            evidence.append(
                ToolEvidence(
                    file_path="changes",
                    content=f"총 {len(output.changes)}개 파일 변경됨",
                    evidence_type="changes",
                    description="감지된 파일 변경사항 요약",
                )
            )

        return evidence

    def _get_error_code(self, error_type: str) -> str:
        """에러 타입에 따른 에러 코드 반환."""
        error_mapping = {
            "INVALID_INPUT": "INVALID_INPUT",
            "PROCESSING_ERROR": "PROCESSING_ERROR",
            "FILE_NOT_FOUND": "FILE_NOT_FOUND",
        }
        return error_mapping.get(error_type, "PROCESSING_ERROR")

    def validate_input(self, input_data: GitAnalysisInput) -> bool:
        """
        입력 데이터 검증.

        Args:
            input_data: 검증할 입력 데이터

        Returns:
            검증 통과 여부
        """
        if not input_data.repo_path:
            return False

        if not Path(input_data.repo_path).exists():
            return False

        return True

    def get_changed_files(self, base_ref: str, head_ref: str) -> list[FileChange]:
        """
        두 Git 참조 간의 모든 변경된 파일을 가져옵니다.

        Args:
            base_ref: 기준 참조 (예: 'main', 'develop', 또는 커밋 해시)
            head_ref: 헤드 참조 (예: 기능 브랜치 또는 커밋 해시)

        Returns:
            모든 변경사항을 나타내는 FileChange 객체들의 리스트

        Raises:
            GitCommandError: Git 명령이 실패한 경우
            ValueError: 참조가 유효하지 않은 경우
        """
        if not self.repo:
            raise ValueError("Git repository not initialized")

        try:
            # 두 참조 간의 diff 가져오기
            base_commit = self.repo.commit(base_ref)
            head_commit = self.repo.commit(head_ref)

            # diff 생성
            diff = base_commit.diff(head_commit)

            # 각 변경사항 처리
            changes = []
            for change in diff:
                file_change = self._process_diff_change(change)
                if file_change:
                    changes.append(file_change)

            logger.info(
                f"{base_ref}와 {head_ref} 간에 {len(changes)}개 파일 변경사항 감지됨"
            )
            return changes

        except GitCommandError as e:
            logger.error(f"Git 명령 실패: {e}")
            raise
        except Exception as e:
            logger.error(f"변경된 파일 가져오기 중 예상치 못한 오류: {e}")
            raise

    def _process_diff_change(self, diff: Diff) -> FileChange | None:
        """
        단일 diff 변경사항을 처리하고 FileChange 객체로 변환합니다.

        Args:
            diff: Git diff 객체

        Returns:
            FileChange 객체 또는 처리할 수 없는 경우 None
        """
        try:
            # 변경 타입과 파일 경로 결정
            if diff.new_file:
                change_type = "A"  # 추가됨
                file_path = diff.b_path or ""
            elif diff.deleted_file:
                change_type = "D"  # 삭제됨
                file_path = diff.a_path or ""
            elif diff.renamed_file:
                change_type = "R"  # 이름변경됨
                file_path = diff.b_path or ""
            else:
                change_type = "M"  # 수정됨
                file_path = diff.b_path or ""

            # 유효한 파일 경로가 없으면 건너뛰기
            if not file_path:
                logger.warning("유효한 파일 경로가 없는 diff 변경사항 건너뛰기")
                return None

            # diff 통계 안전하게 가져오기
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

            # diff 내용 안전하게 가져오기
            try:
                diff_content = (
                    diff.diff.decode("utf-8", errors="ignore") if diff.diff else ""
                )
            except (AttributeError, UnicodeDecodeError):
                diff_content = ""

            # 이름변경된 파일의 유사도 계산
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
            logger.warning(f"diff 변경사항 처리 실패: {e}")
            return None

    def get_commit_info(self, commit_ref: str) -> CommitInfo:
        """
        특정 커밋에 대한 상세 정보를 가져옵니다.

        Args:
            commit_ref: 커밋 참조 (해시, 브랜치 이름 등)

        Returns:
            커밋 메타데이터가 포함된 CommitInfo 객체

        Raises:
            GitCommandError: Git 명령이 실패한 경우
        """
        if not self.repo:
            raise ValueError("Git repository not initialized")

        try:
            commit = self.repo.commit(commit_ref)

            # 이 커밋에서 변경된 파일들 가져오기
            files_changed = []
            if commit.parents:
                # 부모 커밋과 비교
                diff = commit.parents[0].diff(commit)
                files_changed = [change.b_path for change in diff if change.b_path]
            else:
                # 루트 커밋 - 모든 파일이 새로 생성됨
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
            logger.error(f"{commit_ref}에 대한 커밋 정보 가져오기 실패: {e}")
            raise

    def _get_repository_stats(self) -> dict[str, Any]:
        """저장소에 대한 일반 통계를 가져옵니다."""
        if not self.repo:
            return {}

        try:
            # 활성 브랜치
            active_branch = self.repo.active_branch.name

            # 총 커밋 수
            total_commits = len(list(self.repo.iter_commits()))

            # 최근 커밋 (지난 30일)
            cutoff_date = datetime.now(UTC) - timedelta(days=30)
            recent_commits = len(
                [
                    commit
                    for commit in self.repo.iter_commits()
                    if commit.authored_datetime.replace(tzinfo=UTC) > cutoff_date
                ]
            )

            # 파일 수
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
            logger.error(f"저장소 통계 가져오기 실패: {e}")
            return {}

    def _get_diff_summary(self, base_ref: str, head_ref: str) -> dict[str, Any]:
        """두 참조 간의 변경사항 요약을 가져옵니다."""
        try:
            changes = self.get_changed_files(base_ref, head_ref)

            # 통계 계산
            total_additions = sum(change.additions for change in changes)
            total_deletions = sum(change.deletions for change in changes)

            # 변경 타입별 그룹화
            change_types: dict[str, list[str]] = {}
            for change in changes:
                change_type = change.change_type
                if change_type not in change_types:
                    change_types[change_type] = []
                change_types[change_type].append(change.file_path)

            # 파일 확장자
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
            logger.error(f"diff 요약 가져오기 실패: {e}")
            return {}

    def _get_branch_summary(self, base_branch: str, head_branch: str) -> dict[str, Any]:
        """두 브랜치 간의 포괄적인 변경사항 요약을 가져옵니다."""
        if not self.repo:
            return {}

        try:
            # 기본 diff 요약 가져오기
            diff_summary = self._get_diff_summary(base_branch, head_branch)

            # 헤드 브랜치에 대한 커밋 정보 가져오기
            head_commit = self.repo.heads[head_branch].commit
            base_commit = self.repo.heads[base_branch].commit

            # 브랜치 간 커밋 수 계산
            commit_count = 0
            for _commit in self.repo.iter_commits(f"{base_branch}..{head_branch}"):
                commit_count += 1

            # 작성자 통계
            authors: dict[str, int] = {}
            for commit in self.repo.iter_commits(f"{base_branch}..{head_branch}"):
                author = commit.author.name or "Unknown"
                authors[author] = authors.get(author, 0) + 1

            # 향상된 요약
            enhanced_summary = {
                **diff_summary,
                "commit_count": commit_count,
                "authors": authors,
                "base_branch": base_branch,
                "head_branch": head_branch,
                "base_commit": base_commit.hexsha[:8],
                "head_commit": head_commit.hexsha[:8],
            }

            logger.info(f"{base_branch} vs {head_branch}에 대한 포괄적인 요약 생성됨")
            return enhanced_summary

        except Exception as e:
            logger.error(f"브랜치 요약 가져오기 실패: {e}")
            return {}


def main() -> None:
    """
    GitChangeDetector 클래스 사용 예시.

    이 스크립트는 GitChangeDetector를 사용하여 다음을 수행하는 방법을 보여줍니다:
    1. Git 저장소에 대한 연결 초기화
    2. 참조(브랜치 또는 커밋) 간의 변경된 파일 가져오기
    3. 커밋 정보 및 변경사항 분석
    4. 포괄적인 변경사항 요약 생성
    5. 커밋별 및 범위 분석 메서드 사용

    사용 예시:
        # 현재 디렉토리에서 실행 (Git 저장소여야 함)
        python git_changes.py

        # 특정 Git 저장소에서 실행
        python git_changes.py /path/to/your/repo

        # 프로젝트 루트에서 실행
        python tools/git/git_changes.py

        # 절대 경로로 실행
        python tools/git/git_changes.py /Users/username/projects/my-repo

    PR과 유사한 브랜치 비교:
        스크립트는 자동으로 현재 브랜치를 감지하고 메인 브랜치(main/master/develop)와
        비교하여 PR에서 볼 수 있는 것과 같은 변경사항을 보여줍니다:

        # feature-branch에 있다면, main과 비교됩니다:
        git checkout feature-branch
        python git_changes.py

        # 이는 feature-branch와 main 간의 모든 변경사항을 보여줍니다
        # Pull Request에서 볼 수 있는 것과 유사합니다

    명령줄 인수:
        repo_path (선택사항): Git 저장소 경로
                             기본값: 현재 디렉토리 (.)

    출력:
        - 저장소 통계 및 메타데이터
        - HEAD와 부모 커밋 간의 변경사항 상세 분석
        - 파일별 변경사항 정보와 diff 미리보기
        - 통계와 세부사항이 포함된 변경사항 요약
        - 현재 HEAD 커밋 정보
        - 샘플 파일 히스토리 (Python 파일이 존재하는 경우)

    요구사항:
        - Git 저장소 내에서 실행되어야 함
        - 저장소는 최소 하나의 커밋이 있어야 함
        - Python 패키지: gitpython, loguru
    """

    # 명령줄 인수로 저장소 경로가 제공되었는지 확인
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        # 기본값: 현재 디렉토리
        repo_path = "."

    try:
        # Git Change Detector 초기화
        logger.info(f"Git Change Detector 초기화 중: {repo_path}")
        detector = GitChangeDetector(repo_path)

        # 저장소 통계 가져오기
        logger.info("저장소 통계 가져오는 중...")
        stats = detector._get_repository_stats()
        logger.info(f"저장소 통계: {stats}")

        # 예시: 현재 브랜치와 메인 브랜치 간의 변경사항 가져오기 (PR과 유사한 시나리오)
        try:
            # 현재 브랜치 이름 가져오기
            if not detector.repo:
                logger.error("Git repository not initialized")
                return
            current_branch = detector.repo.active_branch.name
            logger.info(f"현재 브랜치: {current_branch}")

            # 메인 브랜치가 존재하는지 확인 (일반적인 이름들 시도)
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
                logger.info(f"{current_branch} 브랜치와 {main_branch} 브랜치 비교 중")

                # 현재 브랜치와 메인 브랜치 간의 변경사항 가져오기
                changes = detector.get_changed_files(main_branch, current_branch)
                logger.info(f"{len(changes)}개 변경된 파일 발견")

                # 상세 변경사항 표시
                for i, change in enumerate(changes, 1):
                    logger.info(f"\n변경사항 {i}:")
                    logger.info(f"  파일: {change.file_path}")
                    logger.info(f"  타입: {change.change_type}")
                    logger.info(f"  추가: {change.additions}")
                    logger.info(f"  삭제: {change.deletions}")

                    if change.change_type == "R" and change.similarity:
                        logger.info(f"  이름변경 유사도: {change.similarity}")

                    # diff 미리보기 표시 (가독성을 위해 잘림)
                    diff_preview = (
                        change.diff_content[:200] + "..."
                        if len(change.diff_content) > 200
                        else change.diff_content
                    )
                    logger.info(f"  Diff 미리보기:\n{diff_preview}")

                # 변경사항 요약 가져오기
                summary = detector._get_diff_summary(main_branch, current_branch)
                logger.info(f"\n변경사항 요약 ({current_branch} vs {main_branch}):")
                logger.info(f"  총 변경된 파일: {summary['total_files_changed']}")
                logger.info(f"  총 추가: {summary['total_additions']}")
                logger.info(f"  총 삭제: {summary['total_deletions']}")
                logger.info(f"  순 변경: {summary['net_change']}")

                # 변경 타입별 세부사항 표시
                logger.info("  변경 타입:")
                for change_type, files in summary["change_types"].items():
                    logger.info(f"    {change_type}: {len(files)}개 파일")

                # 파일 확장자별 세부사항 표시
                logger.info("  파일 확장자:")
                for ext, count in summary["file_extensions"].items():
                    logger.info(f"    {ext or '확장자 없음'}: {count}개 파일")

            elif main_branch == current_branch:
                logger.info(f"현재 {main_branch} 브랜치에 있음 - 비교 불필요")

                # 대안: 부모 커밋과 비교
                head_commit = detector.repo.head.commit
                if head_commit.parents:
                    base_ref = head_commit.parents[0].hexsha
                    head_ref = head_commit.hexsha

                    logger.info(
                        f"부모 커밋 비교로 대체: {base_ref[:8]} vs {head_ref[:8]}"
                    )
                    changes = detector.get_changed_files(base_ref, head_ref)
                    logger.info(f"마지막 커밋에서 {len(changes)}개 변경된 파일 발견")

            else:
                logger.warning(
                    f"메인 브랜치를 찾을 수 없음. 사용 가능한 브랜치: {[b.name for b in detector.repo.heads]}"
                )

        except Exception as e:
            logger.error(f"브랜치 변경사항 분석 중 오류: {e}")

        # 예시: 현재 HEAD에 대한 커밋 정보 가져오기
        try:
            if not detector.repo:
                logger.error("Git repository not initialized")
                return
            head_commit = detector.repo.head.commit
            commit_info = detector.get_commit_info(head_commit.hexsha)

            logger.info("\n현재 HEAD 커밋 정보:")
            logger.info(f"  해시: {commit_info.commit_hash}")
            logger.info(f"  작성자: {commit_info.author} <{commit_info.author_email}>")
            logger.info(f"  날짜: {commit_info.commit_date}")
            logger.info(f"  메시지: {commit_info.commit_message}")
            logger.info(f"  변경된 파일: {len(commit_info.files_changed)}개")

        except Exception as e:
            logger.error(f"커밋 정보 가져오기 오류: {e}")

        # 예시: 새로운 커밋별 메서드 시연
        try:
            if not detector.repo:
                logger.error("Git repository not initialized")
                return
            head_commit = detector.repo.head.commit

            if head_commit.parents:
                # 상세 커밋 정보는 현재 구현되지 않음
                logger.info(f"\n{head_commit.hexsha[:8]}에 대한 커밋 정보:")
                logger.info(f"  커밋 해시: {head_commit.hexsha}")
                logger.info(f"  커밋 메시지: {head_commit.message}")

        except Exception as e:
            logger.error(f"상세 커밋 정보 가져오기 오류: {e}")

        # 예시: 브랜치 요약 메서드 시연
        try:
            if not detector.repo:
                logger.error("Git repository not initialized")
                return
            current_branch = detector.repo.active_branch.name

            # 비교를 위한 메인 브랜치 찾기
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
                logger.info("\n포괄적인 브랜치 요약 생성 중...")
                branch_summary = detector._get_branch_summary(
                    main_branch, current_branch
                )

                if branch_summary:
                    logger.info(f"브랜치 요약 ({current_branch} vs {main_branch}):")
                    logger.info(f"  총 커밋: {branch_summary.get('commit_count', 0)}")
                    logger.info(f"  작성자: {branch_summary.get('authors', {})}")
                    logger.info(
                        f"  기준 커밋: {branch_summary.get('base_commit', 'N/A')}"
                    )
                    logger.info(
                        f"  헤드 커밋: {branch_summary.get('head_commit', 'N/A')}"
                    )

        except Exception as e:
            logger.error(f"브랜치 요약 생성 오류: {e}")

        # 예시: 특정 파일의 히스토리 가져오기 (존재하는 경우)
        try:
            if not detector.repo:
                logger.error("Git repository not initialized")
                return
            # 저장소에서 Python 파일 찾기
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
                # 발견된 첫 번째 Python 파일 정보 표시
                sample_file = python_files[0]
                logger.info(f"\n샘플 Python 파일: {sample_file}")
                logger.info("파일 히스토리 기능은 현재 구현되지 않음")

        except Exception as e:
            logger.error(f"파일 히스토리 가져오기 오류: {e}")

    except Exception as e:
        logger.error(f"Git Change Detector 초기화 실패: {e}")
        logger.error(
            "유효한 Git 저장소 내에서 실행되거나 유효한 경로를 제공해야 합니다"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
