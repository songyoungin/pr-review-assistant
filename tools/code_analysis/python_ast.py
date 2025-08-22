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
from typing import Any, cast

from loguru import logger


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
        """Initialize the AST node visitor."""
        self.functions: list[FunctionInfo] = []
        self.classes: list[ClassInfo] = []
        self.imports: list[ImportInfo] = []
        self.current_nesting = 0
        self.magic_numbers: list[tuple[int, str]] = []
        self.hardcoded_strings: list[tuple[int, str]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition nodes."""
        # Calculate complexity (simplified cyclomatic complexity)
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Extract arguments
        arguments = [arg.arg for arg in node.args.args]

        # Extract decorators
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

        # Extract docstring
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
        """Visit async function definition nodes."""
        # Calculate complexity (simplified cyclomatic complexity)
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, ast.If | ast.While | ast.For | ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        # Extract arguments
        arguments = [arg.arg for arg in node.args.args]

        # Extract decorators
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

        # Extract docstring
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
        """Visit class definition nodes."""
        # Extract base classes
        bases: list[str] = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                if isinstance(base.value, ast.Name):
                    bases.append(f"{base.value.id}.{base.attr}")
                else:
                    bases.append(f"<complex>.{base.attr}")

        # Extract decorators
        decorators: list[str] = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                if isinstance(decorator.value, ast.Name):
                    decorators.append(f"{decorator.value.id}.{decorator.attr}")
                else:
                    decorators.append(f"<complex>.{decorator.attr}")

        # Extract docstring
        docstring = None
        if (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            docstring = node.body[0].value.value

        # Extract class variables
        class_variables: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        class_variables.append(target.id)

        # Create class info
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

        # Filter methods that belong to this class
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
        """Visit import nodes."""
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
        """Visit from-import nodes."""
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
        """Visit number nodes to detect magic numbers."""
        # Consider numbers other than 0, 1, -1 as potential magic numbers
        if node.n not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.n)))
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        """Visit constant nodes to detect magic numbers and hardcoded strings."""
        if isinstance(node.value, int | float) and node.value not in (0, 1, -1):
            self.magic_numbers.append((node.lineno, str(node.value)))
        elif isinstance(node.value, str) and len(node.value) > 20:
            # Consider long strings as potentially hardcoded
            self.hardcoded_strings.append((node.lineno, node.value[:50] + "..."))
        self.generic_visit(node)


class PythonASTAnalyzer:
    """
    A comprehensive tool for analyzing Python code using AST.

    This class provides methods to:
    1. Parse Python code and build AST representations
    2. Analyze code structure and extract function/class information
    3. Calculate code complexity and quality metrics
    4. Detect potential code smells and issues
    5. Generate comprehensive code analysis reports
    """

    def __init__(self) -> None:
        """Initialize the Python AST Analyzer."""
        self.visitor = ASTNodeVisitor()
        self.ast_tree: ast.AST | None = None
        self.source_code: str = ""
        self.file_path: Path | None = None

    def analyze_file(self, file_path: str | Path) -> dict[str, Any]:
        """
        Analyze a Python file and return comprehensive analysis results.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            FileNotFoundError: If the file doesn't exist
            SyntaxError: If the file contains syntax errors
            ValueError: If the file is not a Python file
        """
        file_path = Path(file_path)

        # Validate file
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if file_path.suffix != ".py":
            raise ValueError(f"File is not a Python file: {file_path}")

        self.file_path = file_path

        # Read and parse the file
        try:
            with open(file_path, encoding="utf-8") as f:
                self.source_code = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, encoding="latin-1") as f:
                self.source_code = f.read()

        return self.analyze_source(self.source_code)

    def analyze_source(self, source_code: str) -> dict[str, Any]:
        """
        Analyze Python source code and return comprehensive analysis results.

        Args:
            source_code: Python source code as string

        Returns:
            Dictionary containing comprehensive analysis results

        Raises:
            SyntaxError: If the source code contains syntax errors
        """
        self.source_code = source_code

        # Reset visitor state
        self.visitor = ASTNodeVisitor()

        try:
            # Parse the source code
            self.ast_tree = ast.parse(source_code)

            # Visit all nodes
            self.visitor.visit(self.ast_tree)

            # Calculate metrics
            metrics = self._calculate_metrics()

            # Generate analysis results
            results = {
                "file_info": {
                    "file_path": (
                        str(self.file_path) if self.file_path else "source_code"
                    ),
                    "total_lines": len(source_code.splitlines()),
                    "analysis_timestamp": self._get_timestamp(),
                },
                "functions": [
                    self._function_to_dict(f) for f in self.visitor.functions
                ],
                "classes": [self._class_to_dict(c) for c in self.visitor.classes],
                "imports": [self._import_to_dict(i) for i in self.visitor.imports],
                "metrics": metrics,
                "code_smells": self._detect_code_smells(),
                "summary": self._generate_summary(),
            }

            logger.info(
                f"Successfully analyzed Python code: {len(self.visitor.functions)} functions, {len(self.visitor.classes)} classes"
            )
            return results

        except SyntaxError as e:
            logger.error(f"Syntax error in Python code: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during AST analysis: {e}")
            raise

    def _calculate_metrics(self) -> CodeMetrics:
        """Calculate comprehensive code metrics."""
        lines = self.source_code.splitlines()
        total_lines = len(lines)

        # Count different types of lines
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
                    "suggestion": "Consider defining as a named constant",
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

        # Missing docstrings
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
        """Generate a summary of the analysis."""
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
        """Get a textual assessment based on quality score."""
        if quality_score >= 90:
            return "Excellent - Code follows best practices"
        elif quality_score >= 80:
            return "Good - Minor improvements could be made"
        elif quality_score >= 70:
            return "Fair - Several areas need attention"
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
                "Reduce nesting levels by extracting helper functions or using early returns"
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
        """Get current timestamp as string."""
        from datetime import datetime

        return datetime.now().isoformat()

    def analyze_directory(self, directory_path: str | Path) -> dict[str, Any]:
        """
        Analyze all Python files in a directory.

        Args:
            directory_path: Path to the directory to analyze

        Returns:
            Dictionary containing analysis results for all Python files
        """
        directory_path = Path(directory_path)

        if not directory_path.exists() or not directory_path.is_dir():
            raise ValueError(f"Invalid directory path: {directory_path}")

        python_files = list(directory_path.glob("**/*.py"))

        if not python_files:
            logger.warning(f"No Python files found in directory: {directory_path}")
            return {"directory": str(directory_path), "files": [], "summary": {}}

        results = {"directory": str(directory_path), "files": [], "summary": {}}

        total_metrics = {
            "total_files": len(python_files),
            "total_functions": 0,
            "total_classes": 0,
            "total_imports": 0,
            "total_complexity": 0.0,
            "max_complexity": 0,
            "max_nesting": 0,
            "average_complexity": 0.0,
        }

        for file_path in python_files:
            try:
                file_analysis = self.analyze_file(file_path)
                cast(list[dict[str, Any]], results["files"]).append(
                    {"file_path": str(file_path), "analysis": file_analysis}
                )

                # Aggregate metrics
                metrics = file_analysis["metrics"]
                total_metrics["total_functions"] += metrics.function_count
                total_metrics["total_classes"] += metrics.class_count
                total_metrics["total_imports"] += metrics.import_count
                total_metrics["total_complexity"] += (
                    metrics.average_complexity * metrics.function_count
                )
                total_metrics["max_complexity"] = max(
                    total_metrics["max_complexity"], int(metrics.max_complexity)
                )
                total_metrics["max_nesting"] = max(
                    total_metrics["max_nesting"], int(metrics.max_nesting)
                )

            except Exception as e:
                logger.error(f"Failed to analyze file {file_path}: {e}")
                cast(list[dict[str, Any]], results["files"]).append(
                    {"file_path": str(file_path), "error": str(e)}
                )

        # Calculate averages
        if total_metrics["total_functions"] > 0:
            total_metrics["average_complexity"] = (
                float(total_metrics["total_complexity"])
                / total_metrics["total_functions"]
            )
        else:
            total_metrics["average_complexity"] = 0.0

        results["summary"] = total_metrics

        logger.info(
            f"Successfully analyzed {len(python_files)} Python files in directory"
        )
        return results


def main() -> None:
    """
    Example usage of PythonASTAnalyzer class.

    This demonstrates how to use the PythonASTAnalyzer to:
    1. Analyze individual Python files
    2. Analyze entire directories
    3. Generate comprehensive code analysis reports
    4. Detect code smells and quality issues
    5. Provide improvement recommendations

    Usage Examples:
        # Analyze current directory (must contain Python files)
        python python_ast.py

        # Analyze a specific Python file
        python python_ast.py /path/to/your/file.py

        # Analyze a directory
        python python_ast.py /path/to/your/directory

        # Run from project root
        python tools/code_analysis/python_ast.py

    Command Line Arguments:
        path (optional): Path to Python file or directory to analyze
                        Default: current directory (.)

    Output:
        - File-by-file analysis results
        - Code structure information (functions, classes, imports)
        - Quality metrics and complexity analysis
        - Code smell detection and recommendations
        - Overall quality assessment and score
    """

    # Check if path is provided as command line argument
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Default to current directory
        target_path = "."

    try:
        # Initialize Python AST Analyzer
        logger.info(f"Initializing Python AST Analyzer for: {target_path}")
        analyzer = PythonASTAnalyzer()

        target_path_obj = Path(target_path)

        if target_path_obj.is_file() and target_path_obj.suffix == ".py":
            # Analyze single Python file
            logger.info(f"Analyzing Python file: {target_path_obj}")
            results = analyzer.analyze_file(target_path_obj)

            # Display results
            _display_file_analysis(results)

        elif target_path_obj.is_dir():
            # Analyze directory
            logger.info(f"Analyzing Python files in directory: {target_path_obj}")
            results = analyzer.analyze_directory(target_path_obj)

            # Display results
            _display_directory_analysis(results)

        else:
            logger.error(
                f"Invalid path: {target_path}. Must be a Python file or directory."
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to analyze Python code: {e}")
        sys.exit(1)


def _display_file_analysis(results: dict[str, Any]) -> None:
    """Display analysis results for a single file."""
    logger.info("\n" + "=" * 60)
    logger.info("PYTHON AST ANALYSIS RESULTS")
    logger.info("=" * 60)

    # File info
    file_info = results["file_info"]
    logger.info(f"File: {file_info['file_path']}")
    logger.info(f"Total Lines: {file_info['total_lines']}")
    logger.info(f"Analysis Time: {file_info['analysis_timestamp']}")

    # Summary
    summary = results["summary"]
    logger.info(f"\nQuality Score: {summary['quality_score']}/100")
    logger.info(f"Assessment: {summary['overall_assessment']}")

    # Statistics
    stats = summary["statistics"]
    logger.info("\nStatistics:")
    logger.info(f"  Functions: {stats['total_functions']}")
    logger.info(f"  Classes: {stats['total_classes']}")
    logger.info(f"  Imports: {stats['total_imports']}")
    logger.info(f"  Average Complexity: {stats['average_complexity']}")
    logger.info(f"  Max Complexity: {stats['max_complexity']}")
    logger.info(f"  Max Nesting: {stats['max_nesting']}")

    # Functions
    if results["functions"]:
        logger.info(f"\nFunctions ({len(results['functions'])}):")
        for func in results["functions"]:
            logger.info(
                f"  {func['name']} (line {func['line_number']}) - Complexity: {func['complexity']}"
            )

    # Classes
    if results["classes"]:
        logger.info(f"\nClasses ({len(results['classes'])}):")
        for cls in results["classes"]:
            logger.info(
                f"  {cls['name']} (line {cls['line_number']}) - Methods: {len(cls['methods'])}"
            )

    # Code smells
    smells = results["code_smells"]
    if any(smells.values()):
        logger.info("\nCode Smells Detected:")
        for smell_type, items in smells.items():
            if items:
                logger.info(
                    f"  {smell_type.replace('_', ' ').title()}: {len(items)} issues"
                )

    # Recommendations
    recommendations = summary["recommendations"]
    if recommendations:
        logger.info("\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")


def _display_directory_analysis(results: dict[str, Any]) -> None:
    """Display analysis results for a directory."""
    logger.info("\n" + "=" * 60)
    logger.info("DIRECTORY ANALYSIS RESULTS")
    logger.info("=" * 60)

    logger.info(f"Directory: {results['directory']}")

    # Summary
    summary = results["summary"]
    logger.info("\nDirectory Summary:")
    logger.info(f"  Total Python Files: {summary['total_files']}")
    logger.info(f"  Total Functions: {summary['total_functions']}")
    logger.info(f"  Total Classes: {summary['total_classes']}")
    logger.info(f"  Total Imports: {summary['total_imports']}")
    logger.info(f"  Average Complexity: {summary['average_complexity']:.2f}")
    logger.info(f"  Max Complexity: {summary['max_complexity']}")
    logger.info(f"  Max Nesting: {summary['max_nesting']}")

    # File results
    logger.info("\nFile Analysis Results:")
    for file_result in results["files"]:
        if "error" in file_result:
            logger.warning(
                f"  {file_result['file_path']}: ERROR - {file_result['error']}"
            )
        else:
            analysis = file_result["analysis"]
            summary = analysis["summary"]
            logger.info(
                f"  {file_result['file_path']}: Score {summary['quality_score']}/100 - {summary['overall_assessment']}"
            )


if __name__ == "__main__":
    main()
