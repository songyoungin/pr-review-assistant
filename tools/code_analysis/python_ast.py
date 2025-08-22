"""
Python AST Analysis Tool

This tool provides comprehensive functionality for analyzing Python code using
Abstract Syntax Trees (AST). It can:
- Parse Python code and build AST representations
- Analyze code structure and complexity
- Extract function/class definitions and relationships
- Calculate code quality metrics
- Detect potential code smells and issues
"""

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from ..base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult


@dataclass
class FunctionInfo:
    """Represents information about a Python function."""

    name: str
    line_number: int
    end_line: int
    arguments: list[str]
    decorators: list[str]
    docstring: str | None
    complexity: int
    nested_level: int
    has_return: bool
    has_yield: bool


@dataclass
class ClassInfo:
    """Represents information about a Python class."""

    name: str
    line_number: int
    end_line: int
    bases: list[str]
    decorators: list[str]
    docstring: str | None
    methods: list[FunctionInfo]
    class_variables: list[str]
    nested_level: int


@dataclass
class ImportInfo:
    """Represents information about Python imports."""

    module: str
    names: list[str]
    alias: str | None
    line_number: int
    import_type: str  # 'import', 'from', 'from_as'


@dataclass
class CodeMetrics:
    """Represents overall code quality metrics."""

    total_lines: int
    code_lines: int
    comment_lines: int
    blank_lines: int
    docstring_lines: int
    function_count: int
    class_count: int
    import_count: int
    average_complexity: float
    max_complexity: int
    max_nesting: int
    magic_numbers: list[tuple[int, str]]
    hardcoded_strings: list[tuple[int, str]]


@dataclass
class ASTAnalysisInput:
    """Input data for AST analysis."""

    file_path: str | None = None
    source_code: str | None = None
    directory_path: str | None = None


@dataclass
class ASTAnalysisOutput:
    """AST analysis results."""

    file_info: dict[str, Any] | None = None
    functions: list[dict[str, Any]] | None = None
    classes: list[dict[str, Any]] | None = None
    imports: list[dict[str, Any]] | None = None
    metrics: CodeMetrics | None = None
    code_smells: dict[str, list[dict[str, Any]]] | None = None
    summary: dict[str, Any] | None = None


class ASTNodeVisitor(ast.NodeVisitor):
    """
    Python 코드 구조를 분석하기 위한 커스텀 AST 노드 방문자.

    이 방문자는 AST를 순회하며 다음에 대한 정보를 수집합니다:
    - 함수와 그들의 복잡성
    - 클래스와 그들의 메서드
    - import와 그들의 구조
    - 코드 메트릭과 품질 지표
    """

    def __init__(self) -> None:
        """AST 노드 방문자 초기화."""
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.current_nesting = 0
        self.magic_numbers: list[tuple[int, str]] = []
        self.hardcoded_strings: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """함수 정의 노드 방문."""
        # 복잡성 계산 (간소화된 순환 복잡성)
        complexity = 1  # 기본 복잡성

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # 인수 추출
        arguments = [arg.arg for arg in node.args.args]

        # 데코레이터 추출
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # return/yield 문 확인
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        # docstring 추출
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        function_info = FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            arguments=arguments,
            decorators=decorators,
            docstring=docstring,
            complexity=complexity,
            nested_level=self.current_nesting,
            has_return=has_return,
            has_yield=has_yield,
        )

        self.functions.append(function_info)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """비동기 함수 정의 노드 방문."""
        # 복잡성 계산 (간소화된 순환 복잡성)
        complexity = 1  # 기본 복잡성

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # 인수 추출
        arguments = [arg.arg for arg in node.args.args]

        # 데코레이터 추출
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # return/yield 문 확인
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        # docstring 추출
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        function_info = FunctionInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            arguments=arguments,
            decorators=decorators,
            docstring=docstring,
            complexity=complexity,
            nested_level=self.current_nesting,
            has_return=has_return,
            has_yield=has_yield,
        )

        self.functions.append(function_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """클래스 정의 노드 방문."""
        # 기본 클래스 추출
        bases: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name):
                    bases.append(f"{base.value.id}.{base.attr}")
                else:
                    bases.append(f"<complex>.{base.attr}")

        # 데코레이터 추출
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # docstring 추출
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        # 클래스 변수 추출
        class_variables: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        # 클래스 정보 생성
        class_info = ClassInfo(
            name=node.name,
            line_number=node.lineno,
            end_line=node.end_lineno or node.lineno,
            bases=bases,
            decorators=decorators,
            docstring=docstring,
            methods=[],
            class_variables=class_variables,
            nested_level=self.current_nesting,
        )

        # 메서드를 찾기 위해 클래스 본문 방문
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1

        # 이 클래스에 속하는 메서드 필터링
        class_methods = [
            func
            for func in self.functions
            if (
                func.line_number > node.lineno
                and func.end_line < (node.end_lineno or node.lineno)
            )
        ]
        class_info.methods = class_methods

        self.classes.append(class_info)

    def visit_Import(self, node: ast.Import) -> None:
        """import 노드 방문."""
        for alias in node.names:
            import_info = ImportInfo(
                module="",
                names=[alias.name],
                alias=alias.asname,
                line_number=node.lineno,
                import_type="import",
            )
            self.imports.append(import_info)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """from-import 노드 방문."""
        module = node.module or ""
        for alias in node.names:
            import_info = ImportInfo(
                module=module,
                names=[alias.name],
                alias=alias.asname,
                line_number=node.lineno,
                import_type="from_as" if alias.asname else "from",
            )
            self.imports.append(import_info)

    def visit_Num(self, node: ast.Num) -> None:
        """숫자 노드 방문하여 매직 넘버 감지."""
        # 0, 1, -1이 아닌 숫자를 잠재적 매직 넘버로 간주
        if node.n not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.n)))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """상수 노드 방문하여 매직 넘버와 하드코딩된 문자열 감지."""
        if isinstance(node.value, int | float) and node.value not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.value)))
        elif isinstance(node.value, str) and len(node.value) > 20:
            # 긴 문자열을 잠재적으로 하드코딩된 것으로 간주
            self.hardcoded_strings.append((node.lineno, node.value[:50] + "..."))
        self.generic_visit(node)


class PythonASTAnalyzer(BaseTool[ASTAnalysisInput, ASTAnalysisOutput]):
    """
    AST를 사용하여 Python 코드를 분석하는 포괄적인 툴.

    이 클래스는 다음 메서드들을 제공합니다:
    1. Python 코드를 파싱하고 AST 표현을 구축
    2. 코드 구조를 분석하고 함수/클래스 정보를 추출
    3. 코드 복잡성과 품질 메트릭을 계산
    4. 잠재적인 코드 냄새와 문제를 감지
    5. 포괄적인 코드 분석 보고서를 생성
    """

    def __init__(self) -> None:
        """Python AST Analyzer 초기화."""
        super().__init__("PythonASTAnalyzer")
        self.visitor = ASTNodeVisitor()
        self.ast_tree: ast.AST | None = None
        self.source_code: str = ""
        self.file_path: Path | None = None

    def execute(self, input_data: ASTAnalysisInput) -> ToolResult[ASTAnalysisOutput]:
        """
        Python 코드 분석 실행.

        Args:
            input_data: AST 분석을 위한 입력 데이터

        Returns:
            AST 분석 결과
        """
        try:
            # 입력 검증
            if not self.validate_input(input_data):
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="잘못된 입력 데이터입니다",
                    metrics=self._create_metrics(),
                )

            # 분석 실행
            if input_data.file_path:
                output = self._analyze_file(input_data.file_path)
            elif input_data.source_code:
                output = self._analyze_source(input_data.source_code)
            elif input_data.directory_path:
                output = self._analyze_directory(input_data.directory_path)
            else:
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="파일 경로, 소스 코드, 또는 디렉토리 경로 중 하나가 필요합니다",
                    metrics=self._create_metrics(),
                )

            # 증거 생성
            evidence = self._create_evidence(input_data, output)

            # 메트릭 생성
            metrics = self._create_metrics(
                files_processed=1 if input_data.file_path else 0,
                lines_processed=len(self.source_code.splitlines())
                if self.source_code
                else 0,
            )

            return ToolResult.success(
                output=output,
                evidence=evidence,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Python AST 분석 실행 중 오류 발생: {e}")
            return ToolResult.error(
                error_code=ToolErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                metrics=self._create_metrics(),
            )

    def validate_input(self, input_data: ASTAnalysisInput) -> bool:
        """
        입력 데이터 검증.

        Args:
            input_data: 검증할 입력 데이터

        Returns:
            검증 통과 여부
        """
        if input_data.file_path and not Path(input_data.file_path).exists():
            return False

        if input_data.directory_path and not Path(input_data.directory_path).is_dir():
            return False

        if not any(
            [input_data.file_path, input_data.source_code, input_data.directory_path]
        ):
            return False

        return True

    def _create_evidence(
        self, input_data: ASTAnalysisInput, output: ASTAnalysisOutput
    ) -> list[ToolEvidence]:
        """분석 결과에 대한 증거 생성."""
        evidence = []

        # 파일 정보 증거
        if input_data.file_path:
            evidence.append(
                ToolEvidence(
                    file_path=input_data.file_path,
                    content=f"Python 파일 분석: {input_data.file_path}",
                    evidence_type="file",
                    description="분석된 Python 파일 경로",
                )
            )

        # 분석 결과 증거
        if output.metrics:
            evidence.append(
                ToolEvidence(
                    file_path="metrics",
                    content=f"함수 {output.metrics.function_count}개, 클래스 {output.metrics.class_count}개 분석됨",
                    evidence_type="analysis",
                    description="코드 분석 결과 요약",
                )
            )

        return evidence

    def _analyze_file(self, file_path: str | Path) -> ASTAnalysisOutput:
        """
        Python 파일을 분석하고 포괄적인 분석 결과를 반환합니다.

        Args:
            file_path: 분석할 Python 파일의 경로

        Returns:
            포괄적인 분석 결과를 담은 딕셔너리

        Raises:
            FileNotFoundError: 파일이 존재하지 않는 경우
            SyntaxError: 파일에 구문 오류가 있는 경우
            ValueError: 파일이 Python 파일이 아닌 경우
        """
        file_path = Path(file_path)

        # 파일 검증
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        if file_path.suffix != ".py":
            raise ValueError(f"파일이 Python 파일이 아닙니다: {file_path}")

        self.file_path = file_path

        # 파일 읽기 및 파싱
        try:
            with open(file_path, encoding="utf-8") as f:
                self.source_code = f.read()
        except UnicodeDecodeError:
            # 다른 인코딩으로 시도
            with open(file_path, encoding="latin-1") as f:
                self.source_code = f.read()

        return self._analyze_source(self.source_code)

    def _analyze_source(self, source_code: str) -> ASTAnalysisOutput:
        """
        Python 소스 코드를 분석하고 포괄적인 분석 결과를 반환합니다.

        Args:
            source_code: 문자열로 된 Python 소스 코드

        Returns:
            포괄적인 분석 결과를 담은 딕셔너리

        Raises:
            SyntaxError: 소스 코드에 구문 오류가 있는 경우
        """
        self.source_code = source_code

        # 방문자 상태 초기화
        self.visitor = ASTNodeVisitor()

        try:
            # 소스 코드 파싱
            self.ast_tree = ast.parse(source_code)

            # 모든 노드 방문
            self.visitor.visit(self.ast_tree)

            # 메트릭 계산
            metrics = self._calculate_metrics()

            # 분석 결과 생성
            results = ASTAnalysisOutput(
                file_info={
                    "file_path": (
                        str(self.file_path) if self.file_path else "source_code"
                    ),
                    "total_lines": len(source_code.splitlines()),
                    "analysis_timestamp": self._get_timestamp(),
                },
                functions=[self._function_to_dict(f) for f in self.visitor.functions],
                classes=[self._class_to_dict(c) for c in self.visitor.classes],
                imports=[self._import_to_dict(i) for i in self.visitor.imports],
                metrics=metrics,
                code_smells=self._detect_code_smells(),
                summary=self._generate_summary(),
            )

            logger.info(
                f"Python 코드 분석 성공: {len(self.visitor.functions)}개 함수, {len(self.visitor.classes)}개 클래스"
            )
            return results

        except SyntaxError as e:
            logger.error(f"Python 코드 구문 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"AST 분석 중 예상치 못한 오류: {e}")
            raise

    def _calculate_metrics(self) -> CodeMetrics:
        """포괄적인 코드 메트릭을 계산합니다."""
        lines = self.source_code.splitlines()
        total_lines = len(lines)

        # 다양한 타입의 라인 수 계산
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        docstring_lines = 0

        in_docstring = False
        docstring_delimiter = None

        for line in lines:
            stripped = line.strip()

            if not stripped:
                blank_lines += 1
            elif stripped.startswith("#"):
                comment_lines += 1
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                if not in_docstring:
                    in_docstring = True
                    docstring_delimiter = stripped[:3]
                    docstring_lines += 1
                elif docstring_delimiter and stripped.endswith(docstring_delimiter):
                    in_docstring = False
                    docstring_delimiter = None
                else:
                    docstring_lines += 1
            elif in_docstring:
                docstring_lines += 1
            else:
                code_lines += 1

        # 복잡성 메트릭 계산
        complexities = [f.complexity for f in self.visitor.functions]
        average_complexity = (
            sum(complexities) / len(complexities) if complexities else 0
        )
        max_complexity = max(complexities) if complexities else 0

        # 중첩 메트릭 계산
        max_nesting = max(
            [f.nested_level for f in self.visitor.functions]
            + [c.nested_level for c in self.visitor.classes],
            default=0,
        )

        return CodeMetrics(
            total_lines=total_lines,
            code_lines=code_lines,
            comment_lines=comment_lines,
            blank_lines=blank_lines,
            docstring_lines=docstring_lines,
            function_count=len(self.visitor.functions),
            class_count=len(self.visitor.classes),
            import_count=len(self.visitor.imports),
            average_complexity=average_complexity,
            max_complexity=max_complexity,
            max_nesting=max_nesting,
            magic_numbers=self.visitor.magic_numbers,
            hardcoded_strings=self.visitor.hardcoded_strings,
        )

    def _detect_code_smells(self) -> dict[str, list[dict[str, Any]]]:
        """잠재적인 코드 냄새와 문제를 감지합니다."""
        smells: dict[str, list[dict[str, Any]]] = {
            "high_complexity": [],
            "long_functions": [],
            "deep_nesting": [],
            "magic_numbers": [],
            "hardcoded_strings": [],
            "missing_docstrings": [],
        }

        # 높은 복잡성 함수 (복잡성 > 10)
        for func in self.visitor.functions:
            if func.complexity > 10:
                smells["high_complexity"].append(
                    {
                        "name": func.name,
                        "line": func.line_number,
                        "complexity": func.complexity,
                        "type": "function",
                    }
                )

        # 긴 함수 (> 50 라인)
        for func in self.visitor.functions:
            if func.end_line - func.line_number > 50:
                smells["long_functions"].append(
                    {
                        "name": func.name,
                        "line": func.line_number,
                        "length": func.end_line - func.line_number,
                        "type": "function",
                    }
                )

        # 깊은 중첩 (> 4 레벨)
        for func in self.visitor.functions:
            if func.nested_level > 4:
                smells["deep_nesting"].append(
                    {
                        "name": func.name,
                        "line": func.line_number,
                        "nesting": func.nested_level,
                        "type": "function",
                    }
                )

        # 매직 넘버
        for line, value in self.visitor.magic_numbers:
            smells["magic_numbers"].append(
                {
                    "line": line,
                    "value": value,
                    "suggestion": "명명된 상수로 정의하는 것을 고려하세요",
                }
            )

        # 하드코딩된 문자열
        for line, value in self.visitor.hardcoded_strings:
            smells["hardcoded_strings"].append(
                {
                    "line": line,
                    "value": value,
                    "suggestion": "설정으로 외부화하는 것을 고려하세요",
                }
            )

        # 누락된 docstring
        for func in self.visitor.functions:
            if not func.docstring:
                smells["missing_docstrings"].append(
                    {"name": func.name, "line": func.line_number, "type": "function"}
                )

        for cls in self.visitor.classes:
            if not cls.docstring:
                smells["missing_docstrings"].append(
                    {"name": cls.name, "line": cls.line_number, "type": "class"}
                )

        return smells

    def _generate_summary(self) -> dict[str, Any]:
        """분석 요약을 생성합니다."""
        metrics = self._calculate_metrics()

        # 품질 점수 계산 (0-100)
        quality_score = 100

        # 다양한 문제에 대한 점수 차감
        if metrics.max_complexity > 10:
            quality_score -= min(20, (metrics.max_complexity - 10) * 2)

        if metrics.max_nesting > 4:
            quality_score -= min(15, (metrics.max_nesting - 4) * 3)

        if len(self.visitor.magic_numbers) > 0:
            quality_score -= min(10, len(self.visitor.magic_numbers))

        if len(self.visitor.hardcoded_strings) > 0:
            quality_score -= min(10, len(self.visitor.hardcoded_strings))

        # 점수가 0 아래로 내려가지 않도록 보장
        quality_score = max(0, quality_score)

        return {
            "quality_score": quality_score,
            "overall_assessment": self._get_assessment(quality_score),
            "recommendations": self._get_recommendations(),
            "statistics": {
                "total_functions": metrics.function_count,
                "total_classes": metrics.class_count,
                "total_imports": metrics.import_count,
                "average_complexity": round(metrics.average_complexity, 2),
                "max_complexity": metrics.max_complexity,
                "max_nesting": metrics.max_nesting,
            },
        }

    def _get_assessment(self, quality_score: int) -> str:
        """품질 점수에 기반한 텍스트 평가를 반환합니다."""
        if quality_score >= 90:
            return "우수 - 코드가 모범 사례를 따릅니다"
        elif quality_score >= 80:
            return "좋음 - 약간의 개선이 가능합니다"
        elif quality_score >= 70:
            return "보통 - 여러 영역에 주의가 필요합니다"
        elif quality_score >= 60:
            return "나쁨 - 상당한 리팩토링이 권장됩니다"
        else:
            return "매우 나쁨 - 주요 리팩토링이 필요합니다"

    def _get_recommendations(self) -> list[str]:
        """코드 개선을 위한 권장사항을 생성합니다."""
        recommendations = []
        metrics = self._calculate_metrics()

        if metrics.max_complexity > 10:
            recommendations.append(
                "복잡한 함수를 더 작고 관리하기 쉬운 조각들로 나누는 것을 고려하세요"
            )

        if metrics.max_nesting > 4:
            recommendations.append(
                "헬퍼 함수를 추출하거나 조기 반환을 사용하여 중첩 레벨을 줄이세요"
            )

        if self.visitor.magic_numbers:
            recommendations.append("가독성을 위해 매직 넘버를 명명된 상수로 대체하세요")

        if self.visitor.hardcoded_strings:
            recommendations.append(
                "하드코딩된 문자열을 설정 파일이나 상수로 외부화하세요"
            )

        if not recommendations:
            recommendations.append(
                "코드가 좋은 관행을 따릅니다. 계속 좋은 작업을 하세요!"
            )

        return recommendations

    def _function_to_dict(self, func: FunctionInfo) -> dict[str, Any]:
        """FunctionInfo를 딕셔너리로 변환합니다."""
        return {
            "name": func.name,
            "line_number": func.line_number,
            "end_line": func.end_line,
            "arguments": func.arguments,
            "decorators": func.decorators,
            "docstring": func.docstring,
            "complexity": func.complexity,
            "nested_level": func.nested_level,
            "has_return": func.has_return,
            "has_yield": func.has_yield,
        }

    def _class_to_dict(self, cls: ClassInfo) -> dict[str, Any]:
        """ClassInfo를 딕셔너리로 변환합니다."""
        return {
            "name": cls.name,
            "line_number": cls.line_number,
            "end_line": cls.end_line,
            "bases": cls.bases,
            "decorators": cls.decorators,
            "docstring": cls.docstring,
            "methods": [self._function_to_dict(m) for m in cls.methods],
            "class_variables": cls.class_variables,
            "nested_level": cls.nested_level,
        }

    def _import_to_dict(self, imp: ImportInfo) -> dict[str, Any]:
        """ImportInfo를 딕셔너리로 변환합니다."""
        return {
            "module": imp.module,
            "names": imp.names,
            "alias": imp.alias,
            "line_number": imp.line_number,
            "import_type": imp.import_type,
        }

    def _get_timestamp(self) -> str:
        """현재 타임스탬프를 문자열로 반환합니다."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _analyze_directory(self, directory_path: str | Path) -> ASTAnalysisOutput:
        """
        디렉토리의 모든 Python 파일을 분석합니다.

        Args:
            directory_path: 분석할 디렉토리의 경로

        Returns:
            모든 Python 파일에 대한 분석 결과
        """
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"잘못된 디렉토리 경로: {directory_path}")

        python_files = list(directory_path.glob("**/*.py"))

        if not python_files:
            logger.warning(
                f"디렉토리에서 Python 파일을 찾을 수 없습니다: {directory_path}"
            )
            return ASTAnalysisOutput(
                file_info={"directory": str(directory_path)},
                functions=[],
                classes=[],
                imports=[],
                metrics=None,
                code_smells={},
                summary={},
            )

        # 디렉토리 분석 결과 생성
        results = ASTAnalysisOutput(
            file_info={"directory": str(directory_path)},
            functions=[],
            classes=[],
            imports=[],
            metrics=None,
            code_smells={},
            summary={},
        )

        # 모든 파일 분석 결과 집계
        all_functions = []
        all_classes = []
        all_imports = []

        for file_path in python_files:
            try:
                file_analysis = self._analyze_file(file_path)

                # 결과 집계
                if file_analysis.functions:
                    all_functions.extend(file_analysis.functions)
                if file_analysis.classes:
                    all_classes.extend(file_analysis.classes)
                if file_analysis.imports:
                    all_imports.extend(file_analysis.imports)

            except Exception as e:
                logger.error(f"파일 {file_path} 분석 실패: {e}")

        # 집계된 결과 설정
        results.functions = all_functions
        results.classes = all_classes
        results.imports = all_imports

        # 전체 메트릭 계산
        if all_functions or all_classes:
            results.metrics = self._calculate_metrics()
            results.code_smells = self._detect_code_smells()
            results.summary = self._generate_summary()

        logger.info(f"디렉토리에서 {len(python_files)}개 Python 파일 분석 성공")
        return results


def main() -> None:
    """
    PythonASTAnalyzer 클래스 사용 예시.

    이 스크립트는 PythonASTAnalyzer를 사용하여 다음을 수행하는 방법을 보여줍니다:
    1. 개별 Python 파일 분석
    2. 전체 디렉토리 분석
    3. 포괄적인 코드 분석 보고서 생성
    4. 코드 냄새와 품질 문제 감지
    5. 개선 권장사항 제공

    사용 예시:
        # 현재 디렉토리에서 실행 (Python 파일이 포함되어야 함)
        python python_ast.py

        # 특정 Python 파일 분석
        python python_ast.py /path/to/your/file.py

        # 디렉토리 분석
        python python_ast.py /path/to/your/directory

        # 프로젝트 루트에서 실행
        python tools/code_analysis/python_ast.py

    명령줄 인수:
        path (선택사항): 분석할 Python 파일 또는 디렉토리 경로
                        기본값: 현재 디렉토리 (.)

    출력:
        - 파일별 분석 결과
        - 코드 구조 정보 (함수, 클래스, import)
        - 품질 메트릭 및 복잡성 분석
        - 코드 냄새 감지 및 권장사항
        - 전체 품질 평가 및 점수
    """

    # 명령줄 인수로 경로가 제공되었는지 확인
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # 기본값: 현재 디렉토리
        target_path = "."

    try:
        # Python AST Analyzer 초기화
        logger.info(f"Python AST Analyzer 초기화 중: {target_path}")
        analyzer = PythonASTAnalyzer()

        target_path_obj = Path(target_path)

        if target_path_obj.is_file() and target_path_obj.suffix == ".py":
            # 단일 Python 파일 분석
            logger.info(f"Python 파일 분석 중: {target_path_obj}")
            input_data = ASTAnalysisInput(file_path=str(target_path_obj))
            result = analyzer.run(input_data)

            if result.status.value == "success" and result.output:
                _display_file_analysis(result.output)
            else:
                logger.error(f"분석 실패: {result.error_message}")

        elif target_path_obj.is_dir():
            # 디렉토리 분석
            logger.info(f"디렉토리의 Python 파일 분석 중: {target_path_obj}")
            input_data = ASTAnalysisInput(directory_path=str(target_path_obj))
            result = analyzer.run(input_data)

            if result.status.value == "success" and result.output:
                _display_directory_analysis(result.output)
            else:
                logger.error(f"분석 실패: {result.error_message}")

        else:
            logger.error(
                f"잘못된 경로: {target_path}. Python 파일 또는 디렉토리여야 합니다."
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Python 코드 분석 실패: {e}")
        sys.exit(1)


def _display_file_analysis(results: ASTAnalysisOutput) -> None:
    """단일 파일에 대한 분석 결과를 표시합니다."""
    logger.info("\n" + "=" * 60)
    logger.info("PYTHON AST 분석 결과")
    logger.info("=" * 60)

    # 파일 정보
    file_info = results.file_info
    if file_info:
        logger.info(f"파일: {file_info.get('file_path', 'N/A')}")
        logger.info(f"총 라인: {file_info.get('total_lines', 'N/A')}")
        logger.info(f"분석 시간: {file_info.get('analysis_timestamp', 'N/A')}")

    # 요약
    summary = results.summary
    if summary:
        logger.info(f"\n품질 점수: {summary.get('quality_score', 'N/A')}/100")
        logger.info(f"평가: {summary.get('overall_assessment', 'N/A')}")

        # 통계
        stats = summary.get("statistics", {})
        logger.info("\n통계:")
        logger.info(f"  함수: {stats.get('total_functions', 'N/A')}")
        logger.info(f"  클래스: {stats.get('total_classes', 'N/A')}")
        logger.info(f"  Import: {stats.get('total_imports', 'N/A')}")
        logger.info(f"  평균 복잡성: {stats.get('average_complexity', 'N/A')}")
        logger.info(f"  최대 복잡성: {stats.get('max_complexity', 'N/A')}")
        logger.info(f"  최대 중첩: {stats.get('max_nesting', 'N/A')}")

    # 함수
    if results.functions:
        logger.info(f"\n함수 ({len(results.functions)}개):")
        for func in results.functions:
            logger.info(
                f"  {func['name']} (라인 {func['line_number']}) - 복잡성: {func['complexity']}"
            )

    # 클래스
    if results.classes:
        logger.info(f"\n클래스 ({len(results.classes)}개):")
        for cls in results.classes:
            logger.info(
                f"  {cls['name']} (라인 {cls['line_number']}) - 메서드: {len(cls['methods'])}개"
            )

    # 코드 냄새
    smells = results.code_smells
    if smells and any(smells.values()):
        logger.info("\n감지된 코드 냄새:")
        for smell_type, items in smells.items():
            if items:
                logger.info(
                    f"  {smell_type.replace('_', ' ').title()}: {len(items)}개 문제"
                )

    # 권장사항
    if summary and "recommendations" in summary:
        recommendations = summary["recommendations"]
        if recommendations:
            logger.info("\n권장사항:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")


def _display_directory_analysis(results: ASTAnalysisOutput) -> None:
    """디렉토리에 대한 분석 결과를 표시합니다."""
    logger.info("\n" + "=" * 60)
    logger.info("디렉토리 분석 결과")
    logger.info("=" * 60)

    file_info = results.file_info
    if file_info:
        logger.info(f"디렉토리: {file_info.get('directory', 'N/A')}")

    # 요약
    summary = results.summary
    if summary:
        logger.info("\n디렉토리 요약:")
        stats = summary.get("statistics", {})
        logger.info(f"  총 함수: {stats.get('total_functions', 'N/A')}")
        logger.info(f"  총 클래스: {stats.get('total_classes', 'N/A')}")
        logger.info(f"  총 Import: {stats.get('total_imports', 'N/A')}")
        logger.info(f"  평균 복잡성: {stats.get('average_complexity', 'N/A')}")
        logger.info(f"  최대 복잡성: {stats.get('max_complexity', 'N/A')}")
        logger.info(f"  최대 중첩: {stats.get('max_nesting', 'N/A')}")

    # 파일 결과
    logger.info("\n파일 분석 결과:")
    if results.functions:
        logger.info(f"  총 {len(results.functions)}개 함수 분석됨")
    if results.classes:
        logger.info(f"  총 {len(results.classes)}개 클래스 분석됨")


if __name__ == "__main__":
    main()
