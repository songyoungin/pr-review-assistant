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

from tools.base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult


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
    Custom AST node visitor for analyzing Python code structure.

    This visitor traverses the AST and collects information about:
    - Functions and their complexity
    - Classes and their methods
    - Imports and their structure
    - Code metrics and quality indicators
    """

    def __init__(self) -> None:
        """Initialize AST node visitor."""
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.current_nesting = 0
        self.magic_numbers: list[tuple[int, str]] = []
        self.hardcoded_strings: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition node."""
        # Complexity calculation (simplified cyclomatic complexity)
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Argument extraction
        arguments = [arg.arg for arg in node.args.args]

        # Decorator extraction
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # Check for return/yield statements
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        # Docstring extraction
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
        """Visit async function definition node."""
        # Complexity calculation (simplified cyclomatic complexity)
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Argument extraction
        arguments = [arg.arg for arg in node.args.args]

        # Decorator extraction
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # Check for return/yield statements
        has_return = any(isinstance(n, ast.Return) for n in ast.walk(node))
        has_yield = any(isinstance(n, ast.Yield) for n in ast.walk(node))

        # Docstring extraction
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
        """Visit class definition node."""
        # Base class extraction
        bases: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name):
                    bases.append(f"{base.value.id}.{base.attr}")
                else:
                    bases.append(f"<complex>.{base.attr}")

        # Decorator extraction
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # Docstring extraction
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        # Class variable extraction
        class_variables: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        # Class information creation
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

        # Visit class body to find methods
        self.current_nesting += 1
        self.generic_visit(node)
        self.current_nesting -= 1

        # Filter methods belonging to this class
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
        """Visit import node."""
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
        """Visit from-import node."""
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
        """Visit number node to detect magic numbers."""
        # Consider numbers other than 0, 1, -1 as potential magic numbers
        if node.n not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.n)))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant node to detect magic numbers and hardcoded strings."""
        if isinstance(node.value, int | float) and node.value not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.value)))
        elif isinstance(node.value, str) and len(node.value) > 20:
            # Consider long strings as potentially hardcoded
            self.hardcoded_strings.append((node.lineno, node.value[:50] + "..."))
        self.generic_visit(node)


class PythonASTAnalyzer(BaseTool[ASTAnalysisInput, ASTAnalysisOutput]):
    """
    Comprehensive tool for analyzing Python code using AST.

    This class provides the following methods:
    1. Parse Python code and build AST representation
    2. Analyze code structure and extract function/class information
    3. Calculate code complexity and quality metrics
    4. Detect potential code smells and issues
    5. Generate comprehensive code analysis reports
    """

    def __init__(self) -> None:
        """Initialize Python AST Analyzer."""
        super().__init__("PythonASTAnalyzer")
        self.visitor = ASTNodeVisitor()
        self.ast_tree: ast.AST | None = None
        self.source_code: str = ""
        self.file_path: Path | None = None

    def execute(self, input_data: ASTAnalysisInput) -> ToolResult[ASTAnalysisOutput]:
        """
        Execute Python AST analysis.

        Args:
            input_data: Input data for AST analysis

        Returns:
            AST analysis result
        """
        try:
            # Synchronous execution path
            if not self.validate_input(input_data):
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="Invalid input data",
                    metrics=self._create_metrics(),
                )

            if input_data.file_path:
                output = self._analyze_file(input_data.file_path)
            elif input_data.source_code:
                output = self._analyze_source(input_data.source_code)
            elif input_data.directory_path:
                output = self._analyze_directory(input_data.directory_path)
            else:
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="Either file path, source code, or directory path is required",
                    metrics=self._create_metrics(),
                )

            evidence = self._create_evidence(input_data, output)
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
            logger.error(f"Error occurred during Python AST analysis execution: {e}")
            return ToolResult.error(
                error_code=ToolErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                metrics=self._create_metrics(),
            )

    def validate_input(self, input_data: ASTAnalysisInput) -> bool:
        """
        Validate input data.

        Args:
            input_data: Input data to validate

        Returns:
            Whether validation passed
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
        """Generate evidence for analysis results."""
        evidence = []

        # File information evidence
        if input_data.file_path:
            evidence.append(
                ToolEvidence(
                    file_path=input_data.file_path,
                    content=f"Python file analysis: {input_data.file_path}",
                    evidence_type="file",
                    description="Path of analyzed Python file",
                )
            )

        # Analysis result evidence
        if output.metrics:
            evidence.append(
                ToolEvidence(
                    file_path="metrics",
                    content=f"Analyzed {output.metrics.function_count} functions, {output.metrics.class_count} classes",
                    evidence_type="analysis",
                    description="Code analysis result summary",
                )
            )

        return evidence

    def _analyze_file(self, file_path: str | Path) -> ASTAnalysisOutput:
        """
        Analyze Python file and return comprehensive analysis results.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            FileNotFoundError: When file does not exist
            SyntaxError: When file has syntax errors
            ValueError: When file is not a Python file
        """
        file_path = Path(file_path)

        # File validation
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix != ".py":
            raise ValueError(f"File is not a Python file: {file_path}")

        self.file_path = file_path

        # File reading and parsing
        try:
            with open(file_path, encoding="utf-8") as f:
                self.source_code = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, encoding="latin-1") as f:
                self.source_code = f.read()

        return self._analyze_source(self.source_code)

    def _analyze_source(self, source_code: str) -> ASTAnalysisOutput:
        """
        Analyze Python source code and return comprehensive analysis results.

        Args:
            source_code: Python source code as string

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            SyntaxError: When source code has syntax errors
        """
        self.source_code = source_code

        # Initialize visitor state
        self.visitor = ASTNodeVisitor()

        try:
            # Parse source code
            self.ast_tree = ast.parse(source_code)

            # Visit all nodes
            self.visitor.visit(self.ast_tree)

            # Calculate metrics
            metrics = self._calculate_metrics()

            # Generate analysis result
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
                f"Python code analysis successful: {len(self.visitor.functions)} functions, {len(self.visitor.classes)} classes"
            )
            return results

        except SyntaxError as e:
            logger.error(f"Python code syntax error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during AST analysis: {e}")
            raise

    def _calculate_metrics(self) -> CodeMetrics:
        """Calculate comprehensive code metrics."""
        lines = self.source_code.splitlines()
        total_lines = len(lines)

        # Calculate various types of line counts
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

        # Calculate complexity metrics
        complexities = [f.complexity for f in self.visitor.functions]
        average_complexity = (
            sum(complexities) / len(complexities) if complexities else 0
        )
        max_complexity = max(complexities) if complexities else 0

        # Calculate nesting metrics
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
        """Detect potential code smells and issues."""
        smells: dict[str, list[dict[str, Any]]] = {
            "high_complexity": [],
            "long_functions": [],
            "deep_nesting": [],
            "magic_numbers": [],
            "hardcoded_strings": [],
            "missing_docstrings": [],
        }

        # High complexity functions (complexity > 10)
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

        # Long functions (> 50 lines)
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

        # Deep nesting (> 4 levels)
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

        # Magic numbers
        for line, value in self.visitor.magic_numbers:
            smells["magic_numbers"].append(
                {
                    "line": line,
                    "value": value,
                    "suggestion": "Consider defining as named constant",
                }
            )

        # Hardcoded strings
        for line, value in self.visitor.hardcoded_strings:
            smells["hardcoded_strings"].append(
                {
                    "line": line,
                    "value": value,
                    "suggestion": "Consider externalizing to configuration",
                }
            )

        # Missing docstring
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
        """Generate analysis summary."""
        metrics = self._calculate_metrics()

        # Calculate quality score (0-100)
        quality_score = 100

        # Deduct points for various issues
        if metrics.max_complexity > 10:
            quality_score -= min(20, (metrics.max_complexity - 10) * 2)

        if metrics.max_nesting > 4:
            quality_score -= min(15, (metrics.max_nesting - 4) * 3)

        if len(self.visitor.magic_numbers) > 0:
            quality_score -= min(10, len(self.visitor.magic_numbers))

        if len(self.visitor.hardcoded_strings) > 0:
            quality_score -= min(10, len(self.visitor.hardcoded_strings))

        # Ensure score doesn't go below 0
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
        """Return text assessment based on quality score."""
        if quality_score >= 90:
            return "Excellent - Code follows best practices"
        elif quality_score >= 80:
            return "Good - Minor improvements possible"
        elif quality_score >= 70:
            return "Fair - Attention needed in several areas"
        elif quality_score >= 60:
            return "Poor - Significant refactoring recommended"
        else:
            return "Very Poor - Major refactoring required"

    def _get_recommendations(self) -> list[str]:
        """Generate recommendations for code improvement."""
        recommendations = []
        metrics = self._calculate_metrics()

        if metrics.max_complexity > 10:
            recommendations.append(
                "Consider breaking down complex functions into smaller, more manageable pieces"
            )

        if metrics.max_nesting > 4:
            recommendations.append(
                "Extract helper functions or use early returns to reduce nesting levels"
            )

        if self.visitor.magic_numbers:
            recommendations.append(
                "Replace magic numbers with named constants for better readability"
            )

        if self.visitor.hardcoded_strings:
            recommendations.append(
                "Externalize hardcoded strings to configuration files or constants"
            )

        if not recommendations:
            recommendations.append(
                "Code follows good practices. Keep up the good work!"
            )

        return recommendations

    def _function_to_dict(self, func: FunctionInfo) -> dict[str, Any]:
        """Convert FunctionInfo to dictionary."""
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
        """Convert ClassInfo to dictionary."""
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
        """Convert ImportInfo to dictionary."""
        return {
            "module": imp.module,
            "names": imp.names,
            "alias": imp.alias,
            "line_number": imp.line_number,
            "import_type": imp.import_type,
        }

    def _get_timestamp(self) -> str:
        """Return current timestamp as string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _analyze_directory(self, directory_path: str | Path) -> ASTAnalysisOutput:
        """
        Analyze all Python files in directory.

        Args:
            directory_path: Path to directory to analyze

        Returns:
            Analysis results for all Python files
        """
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        python_files = list(directory_path.glob("**/*.py"))

        if not python_files:
            logger.warning(f"No Python files found in directory: {directory_path}")
            return ASTAnalysisOutput(
                file_info={"directory": str(directory_path)},
                functions=[],
                classes=[],
                imports=[],
                metrics=None,
                code_smells={},
                summary={},
            )

        # Generate directory analysis result
        results = ASTAnalysisOutput(
            file_info={"directory": str(directory_path)},
            functions=[],
            classes=[],
            imports=[],
            metrics=None,
            code_smells={},
            summary={},
        )

        # Aggregate all file analysis results
        all_functions = []
        all_classes = []
        all_imports = []

        for file_path in python_files:
            try:
                file_analysis = self._analyze_file(file_path)

                # Aggregate results
                if file_analysis.functions:
                    all_functions.extend(file_analysis.functions)
                if file_analysis.classes:
                    all_classes.extend(file_analysis.classes)
                if file_analysis.imports:
                    all_imports.extend(file_analysis.imports)

            except Exception as e:
                logger.error(f"Failed to analyze file {file_path}: {e}")

        # Set aggregated results
        results.functions = all_functions
        results.classes = all_classes
        results.imports = all_imports

        # Calculate overall metrics
        if all_functions or all_classes:
            results.metrics = self._calculate_metrics()
            results.code_smells = self._detect_code_smells()
            results.summary = self._generate_summary()

        logger.info(
            f"Successfully analyzed {len(python_files)} Python files in directory"
        )
        return results


def main() -> None:
    """
    PythonASTAnalyzer class usage example.

    This script demonstrates how to use PythonASTAnalyzer to:
    1. Analyze individual Python files
    2. Analyze entire directories
    3. Generate comprehensive code analysis reports
    4. Detect code smells and quality issues
    5. Provide improvement recommendations

    Usage examples:
        # Run from current directory (must contain Python files)
        python python_ast.py

        # Analyze specific Python file
        python python_ast.py /path/to/your/file.py

        # Directory analysis
        python python_ast.py /path/to/your/directory

        # Run from project root
        python tools/code_analysis/python_ast.py

    Command line arguments:
        path (optional): Path to Python file or directory to analyze
                        Default: current directory (.)

    Output:
        - File-by-file analysis results
        - Code structure information (functions, classes, imports)
        - Quality metrics and complexity analysis
        - Code smell detection and recommendations
        - Overall quality assessment and scoring
    """

    # Check if path is provided as command line argument
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default: current directory
        target_path = "."

    try:
        # Initialize Python AST Analyzer
        logger.info(f"Initializing Python AST Analyzer: {target_path}")
        analyzer = PythonASTAnalyzer()

        target_path_obj = Path(target_path)

        if target_path_obj.is_file() and target_path_obj.suffix == ".py":
            # Analyze single Python file
            logger.info(f"Analyzing Python file: {target_path_obj}")
            input_data = ASTAnalysisInput(file_path=str(target_path_obj))
            result = analyzer.execute(input_data)

            if result.status.value == "success" and result.output:
                _display_file_analysis(result.output)
            else:
                logger.error(f"Analysis failed: {result.error_message}")

        elif target_path_obj.is_dir():
            # Directory analysis
            logger.info(f"Analyzing Python files in directory: {target_path_obj}")
            input_data = ASTAnalysisInput(directory_path=str(target_path_obj))
            result = analyzer.execute(input_data)

            if result.status.value == "success" and result.output:
                _display_directory_analysis(result.output)
            else:
                logger.error(f"Analysis failed: {result.error_message}")

        else:
            logger.error(
                f"Invalid path: {target_path}. Must be a Python file or directory."
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Python code analysis failed: {e}")
        sys.exit(1)


def _display_file_analysis(results: ASTAnalysisOutput) -> None:
    """Display analysis results for a single file."""
    logger.info("\n" + "=" * 60)
    logger.info("PYTHON AST ANALYSIS RESULTS")
    logger.info("=" * 60)

    # File information
    file_info = results.file_info
    if file_info:
        logger.info(f"File: {file_info.get('file_path', 'N/A')}")
        logger.info(f"Total lines: {file_info.get('total_lines', 'N/A')}")
        logger.info(f"Analysis time: {file_info.get('analysis_timestamp', 'N/A')}")

    # Summary
    summary = results.summary
    if summary:
        logger.info(f"\nQuality score: {summary.get('quality_score', 'N/A')}/100")
        logger.info(f"Assessment: {summary.get('overall_assessment', 'N/A')}")

        # Statistics
        stats = summary.get("statistics", {})
        logger.info("\nStatistics:")
        logger.info(f"  Functions: {stats.get('total_functions', 'N/A')}")
        logger.info(f"  Classes: {stats.get('total_classes', 'N/A')}")
        logger.info(f"  Imports: {stats.get('total_imports', 'N/A')}")
        logger.info(f"  Average complexity: {stats.get('average_complexity', 'N/A')}")
        logger.info(f"  Max complexity: {stats.get('max_complexity', 'N/A')}")
        logger.info(f"  Max nesting: {stats.get('max_nesting', 'N/A')}")

    # Functions
    if results.functions:
        logger.info(f"\nFunctions ({len(results.functions)} total):")
        for func in results.functions:
            logger.info(
                f"  {func['name']} (line {func['line_number']}) - complexity: {func['complexity']}"
            )

    # Classes
    if results.classes:
        logger.info(f"\nClasses ({len(results.classes)} total):")
        for cls in results.classes:
            logger.info(
                f"  {cls['name']} (line {cls['line_number']}) - methods: {len(cls['methods'])} total"
            )

    # Code smells
    smells = results.code_smells
    if smells and any(smells.values()):
        logger.info("\nDetected code smells:")
        for smell_type, items in smells.items():
            if items:
                logger.info(
                    f"  {smell_type.replace('_', ' ').title()}: {len(items)} issues"
                )

    # Recommendations
    if summary and "recommendations" in summary:
        recommendations = summary["recommendations"]
        if recommendations:
            logger.info("\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")


def _display_directory_analysis(results: ASTAnalysisOutput) -> None:
    """Display analysis results for a directory."""
    logger.info("\n" + "=" * 60)
    logger.info("DIRECTORY ANALYSIS RESULTS")
    logger.info("=" * 60)

    file_info = results.file_info
    if file_info:
        logger.info(f"Directory: {file_info.get('directory', 'N/A')}")

    # Summary
    summary = results.summary
    if summary:
        logger.info("\nDirectory summary:")
        stats = summary.get("statistics", {})
        logger.info(f"  Total functions: {stats.get('total_functions', 'N/A')}")
        logger.info(f"  Total classes: {stats.get('total_classes', 'N/A')}")
        logger.info(f"  Total imports: {stats.get('total_imports', 'N/A')}")
        logger.info(f"  Average complexity: {stats.get('average_complexity', 'N/A')}")
        logger.info(f"  Max complexity: {stats.get('max_complexity', 'N/A')}")
        logger.info(f"  Max nesting: {stats.get('max_nesting', 'N/A')}")

    # File results
    logger.info("\nFile analysis results:")
    if results.functions:
        logger.info(f"  Total {len(results.functions)} functions analyzed")
    if results.classes:
        logger.info(f"  Total {len(results.classes)} classes analyzed")


if __name__ == "__main__":
    main()
