"""
Code Documentation Matcher Tool

This tool provides comprehensive functionality for comparing code changes with documentation
and detecting mismatches between implementation and docs. It can:
- Analyze README files and detect inconsistencies with code
- Extract API specifications from code and compare with documentation
- Identify missing or outdated documentation
- Suggest specific updates and improvements
- Generate comprehensive documentation gap reports
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class DocumentationGap:
    """Represents a gap or inconsistency between code and documentation."""

    gap_type: str  # 'missing_doc', 'outdated_doc', 'api_mismatch', 'example_mismatch'
    severity: str  # 'high', 'medium', 'low'
    file_path: str
    line_number: int | None
    description: str
    suggestion: str
    code_context: str | None = None
    doc_context: str | None = None


@dataclass
class APISpecification:
    """Represents an API endpoint specification extracted from code."""

    endpoint: str
    method: str
    function_name: str
    file_path: str
    line_number: int
    parameters: list[str]
    return_type: str | None
    docstring: str | None
    decorators: list[str]


@dataclass
class DocumentationSection:
    """Represents a section of documentation."""

    title: str
    content: str
    file_path: str
    line_number: int
    section_type: str  # 'api', 'usage', 'example', 'installation'


class CodeDocMatcher:
    """
    A comprehensive tool for matching code changes with documentation.

    This class provides methods to:
    1. Compare code changes with documentation
    2. Detect mismatches between implementation and docs
    3. Extract API specifications from code
    4. Analyze README files for completeness
    5. Suggest documentation updates
    6. Generate gap analysis reports
    """

    def __init__(self) -> None:
        """Initialize the Code Documentation Matcher."""
        self.api_specs: list[APISpecification] = []
        self.doc_sections: list[DocumentationSection] = []
        self.gaps: list[DocumentationGap] = []

        # Common patterns for API detection
        self.api_patterns = {
            "fastapi": [
                r'@app\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                r'@router\.(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
            ],
            "flask": [
                r'@app\.(route|get|post|put|delete)\(["\']([^"\']+)["\']',
                r'@blueprint\.(route|get|post|put|delete)\(["\']([^"\']+)["\']',
            ],
            "django": [r'path\(["\']([^"\']+)["\']', r'url\(["\']([^"\']+)["\']'],
            "custom": [
                r'@app_(get|post|put|delete|patch)\(["\']([^"\']+)["\']',
                r'@flask_route\(["\']([^"\']+)["\']',
            ],
        }

    def analyze_project(self, project_path: str) -> dict[str, Any]:
        """
        Analyze the entire project for documentation gaps.

        Args:
            project_path: Path to the project root

        Returns:
            Dictionary containing comprehensive analysis results
        """
        project_path_obj = Path(project_path)

        logger.info(f"Starting comprehensive project analysis: {project_path_obj}")

        # Extract API specifications from code
        self._extract_api_specifications(project_path_obj)

        # Analyze README and documentation files
        self._analyze_documentation(project_path_obj)

        # Detect gaps and inconsistencies
        self._detect_documentation_gaps()

        # Generate comprehensive report
        report = self._generate_analysis_report()

        logger.info(f"Analysis complete. Found {len(self.gaps)} documentation gaps")
        return report

    def _extract_api_specifications(self, project_path: Path) -> None:
        """Extract API specifications from Python code files."""
        python_files = list(project_path.glob("**/*.py"))

        for file_path in python_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()

                # Parse AST for function definitions
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for API decorators
                        api_spec = self._extract_api_from_function(
                            node, file_path, content
                        )
                        if api_spec:
                            self.api_specs.append(api_spec)

            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")

    def _extract_api_from_function(
        self, node: ast.FunctionDef, file_path: Path, content: str
    ) -> APISpecification | None:
        """Extract API specification from a function definition."""
        # Check for decorators that might indicate API endpoints
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                # Look for common API patterns
                for _framework, patterns in self.api_patterns.items():
                    for pattern in patterns:
                        match = re.search(
                            pattern, content[node.lineno - 1 : node.lineno + 10]
                        )
                        if match:
                            endpoint = (
                                match.group(2)
                                if len(match.groups()) > 1
                                else match.group(1)
                            )
                            method = self._extract_http_method(decorator, content)

                            # Extract parameters
                            parameters = [arg.arg for arg in node.args.args]

                            # Extract return type annotation
                            return_type = None
                            if node.returns:
                                if isinstance(node.returns, ast.Name):
                                    return_type = node.returns.id
                                elif isinstance(node.returns, ast.Constant):
                                    return_type = str(node.returns.value)

                            # Extract docstring
                            docstring = ast.get_docstring(node)

                            # Extract decorator names
                            decorator_names = []
                            for dec in node.decorator_list:
                                if isinstance(dec, ast.Name):
                                    decorator_names.append(dec.id)
                                elif isinstance(dec, ast.Attribute):
                                    if isinstance(dec.value, ast.Name):
                                        decorator_names.append(
                                            f"{dec.value.id}.{dec.attr}"
                                        )
                                    else:
                                        decorator_names.append(f"<complex>.{dec.attr}")

                            return APISpecification(
                                endpoint=endpoint,
                                method=method,
                                function_name=node.name,
                                file_path=str(file_path),
                                line_number=node.lineno,
                                parameters=parameters,
                                return_type=return_type,
                                docstring=docstring,
                                decorators=decorator_names,
                            )

        return None

    def _extract_http_method(self, decorator: ast.Call, content: str) -> str:
        """Extract HTTP method from decorator."""
        decorator_str = ast.unparse(decorator)

        if "get(" in decorator_str.lower():
            return "GET"
        elif "post(" in decorator_str.lower():
            return "POST"
        elif "put(" in decorator_str.lower():
            return "PUT"
        elif "delete(" in decorator_str.lower():
            return "DELETE"
        elif "patch(" in decorator_str.lower():
            return "PATCH"
        else:
            return "UNKNOWN"

    def _analyze_documentation(self, project_path: Path) -> None:
        """Analyze README and documentation files."""
        # Look for common documentation files
        doc_files = [
            project_path / "README.md",
            project_path / "docs" / "README.md",
            project_path / "API.md",
            project_path / "docs" / "API.md",
        ]

        for doc_file in doc_files:
            if doc_file.exists():
                self._parse_documentation_file(doc_file)

    def _parse_documentation_file(self, doc_file: Path) -> None:
        """Parse a documentation file and extract sections."""
        try:
            with open(doc_file, encoding="utf-8") as f:
                content = f.read()

            # Split into sections based on headers
            lines = content.split("\n")
            current_section = None
            current_content: list[str] = []

            for i, line in enumerate(lines):
                if line.startswith("#"):
                    # Save previous section
                    if current_section:
                        self.doc_sections.append(
                            DocumentationSection(
                                title=current_section,
                                content="\n".join(current_content).strip(),
                                file_path=str(doc_file),
                                line_number=i + 1,
                                section_type=self._classify_section(current_section),
                            )
                        )

                    # Start new section
                    current_section = line.strip("#").strip()
                    current_content = []
                else:
                    if current_section:
                        current_content.append(line)

            # Save last section
            if current_section:
                self.doc_sections.append(
                    DocumentationSection(
                        title=current_section,
                        content="\n".join(current_content).strip(),
                        file_path=str(doc_file),
                        line_number=len(lines),
                        section_type=self._classify_section(current_section),
                    )
                )

        except Exception as e:
            logger.warning(f"Failed to parse documentation file {doc_file}: {e}")

    def _classify_section(self, title: str) -> str:
        """Classify a documentation section based on its title."""
        title_lower = title.lower()

        if any(word in title_lower for word in ["api", "endpoint", "route"]):
            return "api"
        elif any(word in title_lower for word in ["usage", "example", "demo"]):
            return "usage"
        elif any(word in title_lower for word in ["install", "setup", "configuration"]):
            return "installation"
        else:
            return "general"

    def _detect_documentation_gaps(self) -> None:
        """Detect gaps and inconsistencies between code and documentation."""
        self.gaps.clear()

        # Check for undocumented API endpoints
        self._check_undocumented_apis()

        # Check for missing examples
        self._check_missing_examples()

        # Check for outdated documentation
        self._check_outdated_documentation()

        # Check for incomplete API documentation
        self._check_incomplete_api_docs()

    def _check_undocumented_apis(self) -> None:
        """Check for API endpoints that lack proper documentation."""
        for api_spec in self.api_specs:
            # Check if API is documented in README
            documented = False
            for doc_section in self.doc_sections:
                if doc_section.section_type == "api":
                    if api_spec.endpoint in doc_section.content:
                        documented = True
                        break

            if not documented:
                self.gaps.append(
                    DocumentationGap(
                        gap_type="missing_doc",
                        severity="high",
                        file_path=api_spec.file_path,
                        line_number=api_spec.line_number,
                        description=f"API endpoint '{api_spec.endpoint}' ({api_spec.method}) is not documented in README",
                        suggestion=f"Add documentation for {api_spec.method} {api_spec.endpoint} endpoint",
                        code_context=f"Function: {api_spec.function_name}",
                        doc_context="README API section",
                    )
                )

            # Check for missing docstrings
            if not api_spec.docstring:
                self.gaps.append(
                    DocumentationGap(
                        gap_type="missing_doc",
                        severity="medium",
                        file_path=api_spec.file_path,
                        line_number=api_spec.line_number,
                        description=f"API function '{api_spec.function_name}' lacks docstring",
                        suggestion=f"Add comprehensive docstring for {api_spec.function_name} function",
                        code_context=f"Function: {api_spec.function_name}",
                        doc_context="Function docstring",
                    )
                )

    def _check_missing_examples(self) -> None:
        """Check for missing usage examples in documentation."""
        # Look for API sections without examples
        for doc_section in self.doc_sections:
            if doc_section.section_type == "api":
                if not self._contains_examples(doc_section.content):
                    self.gaps.append(
                        DocumentationGap(
                            gap_type="missing_doc",
                            severity="medium",
                            file_path=doc_section.file_path,
                            line_number=doc_section.line_number,
                            description=f"API section '{doc_section.title}' lacks usage examples",
                            suggestion=f"Add code examples for {doc_section.title}",
                            code_context="API documentation",
                            doc_context=doc_section.title,
                        )
                    )

    def _contains_examples(self, content: str) -> bool:
        """Check if content contains code examples."""
        # Look for code blocks, function calls, or import statements
        example_patterns = [
            r"```python",
            r"```bash",
            r"```json",
            r"import\s+",
            r"from\s+",
            r"def\s+",
            r"class\s+",
            r"\.\w+\(",  # Method calls
        ]

        for pattern in example_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return True

        return False

    def _check_outdated_documentation(self) -> None:
        """Check for potentially outdated documentation."""
        # This is a simplified check - in a real implementation,
        # you might compare timestamps or use git history
        for doc_section in self.doc_sections:
            if doc_section.section_type == "api":
                # Check if documented APIs still exist in code
                if not self._api_still_exists(doc_section):
                    self.gaps.append(
                        DocumentationGap(
                            gap_type="outdated_doc",
                            severity="medium",
                            file_path=doc_section.file_path,
                            line_number=doc_section.line_number,
                            description=f"API section '{doc_section.title}' may be outdated",
                            suggestion=f"Verify and update {doc_section.title} documentation",
                            code_context="API documentation",
                            doc_context=doc_section.title,
                        )
                    )

    def _api_still_exists(self, doc_section: DocumentationSection) -> bool:
        """Check if documented APIs still exist in current code."""
        # Extract potential API endpoints from documentation
        # This is a simplified check
        return True  # Placeholder

    def _check_incomplete_api_docs(self) -> None:
        """Check for incomplete API documentation."""
        for api_spec in self.api_specs:
            if api_spec.docstring:
                # Check docstring quality
                if len(api_spec.docstring) < 20:
                    self.gaps.append(
                        DocumentationGap(
                            gap_type="missing_doc",
                            severity="low",
                            file_path=api_spec.file_path,
                            line_number=api_spec.line_number,
                            description=f"API function '{api_spec.function_name}' has minimal docstring",
                            suggestion=f"Expand docstring for {api_spec.function_name} with parameters and return value details",
                            code_context=f"Function: {api_spec.function_name}",
                            doc_context="Function docstring",
                        )
                    )

    def _generate_analysis_report(self) -> dict[str, Any]:
        """Generate a comprehensive analysis report."""
        # Count gaps by severity
        high_gaps = [g for g in self.gaps if g.severity == "high"]
        medium_gaps = [g for g in self.gaps if g.severity == "medium"]
        low_gaps = [g for g in self.gaps if g.severity == "low"]

        # Count gaps by type
        gap_types: dict[str, int] = {}
        for gap in self.gaps:
            gap_types[gap.gap_type] = gap_types.get(gap.gap_type, 0) + 1

        # Calculate documentation coverage
        total_apis = len(self.api_specs)
        documented_apis = len([api for api in self.api_specs if api.docstring])
        coverage_percentage = (
            (documented_apis / total_apis * 100) if total_apis > 0 else 0
        )

        return {
            "summary": {
                "total_gaps": len(self.gaps),
                "high_priority_gaps": len(high_gaps),
                "medium_priority_gaps": len(medium_gaps),
                "low_priority_gaps": len(low_gaps),
                "api_coverage_percentage": round(coverage_percentage, 2),
                "total_api_endpoints": total_apis,
                "documented_endpoints": documented_apis,
            },
            "gap_breakdown": gap_types,
            "gaps": [self._gap_to_dict(gap) for gap in self.gaps],
            "api_specifications": [
                self._api_spec_to_dict(api) for api in self.api_specs
            ],
            "documentation_sections": [
                self._doc_section_to_dict(section) for section in self.doc_sections
            ],
            "recommendations": self._generate_recommendations(),
        }

    def _gap_to_dict(self, gap: DocumentationGap) -> dict[str, Any]:
        """Convert DocumentationGap to dictionary."""
        return {
            "gap_type": gap.gap_type,
            "severity": gap.severity,
            "file_path": gap.file_path,
            "line_number": gap.line_number,
            "description": gap.description,
            "suggestion": gap.suggestion,
            "code_context": gap.code_context,
            "doc_context": gap.doc_context,
        }

    def _api_spec_to_dict(self, api: APISpecification) -> dict[str, Any]:
        """Convert APISpecification to dictionary."""
        return {
            "endpoint": api.endpoint,
            "method": api.method,
            "function_name": api.function_name,
            "file_path": api.file_path,
            "line_number": api.line_number,
            "parameters": api.parameters,
            "return_type": api.return_type,
            "docstring": api.docstring,
            "decorators": api.decorators,
        }

    def _doc_section_to_dict(self, section: DocumentationSection) -> dict[str, Any]:
        """Convert DocumentationSection to dictionary."""
        return {
            "title": section.title,
            "content": (
                section.content[:200] + "..."
                if len(section.content) > 200
                else section.content
            ),
            "file_path": section.file_path,
            "line_number": section.line_number,
            "section_type": section.section_type,
        }

    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        if len([g for g in self.gaps if g.severity == "high"]) > 0:
            recommendations.append(
                "Address high-priority documentation gaps immediately to improve API usability"
            )

        if len([g for g in self.gaps if g.gap_type == "missing_doc"]) > 0:
            recommendations.append("Add comprehensive docstrings to all API functions")

        if (
            len(
                [
                    g
                    for g in self.gaps
                    if g.gap_type == "missing_doc" and "README" in g.description
                ]
            )
            > 0
        ):
            recommendations.append(
                "Update README.md to include all API endpoints with examples"
            )

        if (
            len(
                [
                    g
                    for g in self.gaps
                    if g.gap_type == "missing_doc"
                    and "example" in g.description.lower()
                ]
            )
            > 0
        ):
            recommendations.append(
                "Add usage examples for all API endpoints in documentation"
            )

        if not recommendations:
            recommendations.append(
                "Documentation is in good shape! Keep up the good work."
            )

        return recommendations

    def match_code_with_docs(
        self, code_changes: list[str], docs: list[str]
    ) -> dict[str, Any]:
        """
        Match code changes with documentation (legacy method for backward compatibility).

        Args:
            code_changes: List of code changes
            docs: List of documentation sections

        Returns:
            Dictionary containing matching results
        """
        # This method is kept for backward compatibility
        # For new functionality, use analyze_project()
        return {
            "matched": [],
            "unmatched": [],
            "suggestions": [],
            "note": "Use analyze_project() for comprehensive analysis",
        }


def main() -> None:
    """
    Example usage of CodeDocMatcher class.

    This demonstrates how to use the CodeDocMatcher to:
    1. Analyze a project for documentation gaps
    2. Extract API specifications from code
    3. Compare code with documentation
    4. Generate comprehensive gap reports
    """

    import sys

    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "."

    try:
        # Initialize the matcher
        matcher = CodeDocMatcher()

        # Analyze the project
        logger.info(f"Analyzing project: {project_path}")
        results = matcher.analyze_project(project_path)

        # Display results
        _display_analysis_results(results)

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


def _display_analysis_results(results: dict[str, Any]) -> None:
    """Display analysis results in a readable format."""
    summary = results["summary"]

    logger.info("\n" + "=" * 60)
    logger.info("DOCUMENTATION GAP ANALYSIS RESULTS")
    logger.info("=" * 60)

    # Summary
    logger.info("\n📊 Summary:")
    logger.info(f"  Total Gaps Found: {summary['total_gaps']}")
    logger.info(f"  High Priority: {summary['high_priority_gaps']}")
    logger.info(f"  Medium Priority: {summary['medium_priority_gaps']}")
    logger.info(f"  Low Priority: {summary['low_priority_gaps']}")
    logger.info(f"  API Coverage: {summary['api_coverage_percentage']}%")
    logger.info(f"  Total API Endpoints: {summary['total_api_endpoints']}")

    # Gap breakdown
    gap_breakdown = results["gap_breakdown"]
    if gap_breakdown:
        logger.info("\n🔍 Gap Breakdown:")
        for gap_type, count in gap_breakdown.items():
            logger.info(f"  {gap_type.replace('_', ' ').title()}: {count}")

    # High priority gaps
    high_gaps = [g for g in results["gaps"] if g["severity"] == "high"]
    if high_gaps:
        logger.info("\n🚨 High Priority Issues:")
        for gap in high_gaps[:5]:  # Show first 5
            logger.info(f"  • {gap['description']}")
            logger.info(f"    Suggestion: {gap['suggestion']}")
            logger.info(f"    File: {gap['file_path']}:{gap['line_number']}")
            logger.info("")

    # Recommendations
    recommendations = results["recommendations"]
    if recommendations:
        logger.info("\n💡 Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
