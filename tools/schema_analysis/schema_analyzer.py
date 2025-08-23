"""
Schema Analysis Tool

This tool provides comprehensive functionality for analyzing database schema changes,
detecting breaking changes, and generating operational guides for deployments.
It can:
- Parse and analyze DDL statements (CREATE, ALTER, DROP)
- Detect breaking changes that could affect applications
- Generate comprehensive operational guides
- Assess migration complexity and impact
- Provide rollback strategies for schema changes
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from loguru import logger

from ..base import BaseTool, ToolErrorCode, ToolEvidence, ToolResult


class ChangeType(Enum):
    """Types of DDL changes that can occur."""

    CREATE = "create"
    ALTER = "alter"
    DROP = "drop"
    RENAME = "rename"
    INDEX = "index"
    CONSTRAINT = "constraint"


class ChangeImpact(Enum):
    """Impact levels for schema changes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MigrationComplexity(Enum):
    """Complexity levels for schema migrations."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class DDLChange:
    """
    Represents a single DDL change operation.

    This structure captures the essential information about a schema change
    including its type, target objects, and potential impact.
    """

    change_type: ChangeType
    target_object: str  # Table, column, index, etc.
    sql_statement: str
    impact: ChangeImpact
    description: str | None = None
    line_number: int | None = None
    file_path: str | None = None
    affected_columns: list[str] = field(default_factory=list)
    constraints_affected: list[str] = field(default_factory=list)
    indexes_affected: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and set defaults after initialization."""
        if not self.description:
            self.description = self._generate_description()

    def _generate_description(self) -> str:
        """Generate a human-readable description of the change."""
        if self.change_type == ChangeType.CREATE:
            return f"Create new {self.target_object}"
        elif self.change_type == ChangeType.ALTER:
            return f"Modify existing {self.target_object}"
        elif self.change_type == ChangeType.DROP:
            return f"Remove {self.target_object}"
        elif self.change_type == ChangeType.RENAME:
            return f"Rename {self.target_object}"
        else:
            return f"Modify {self.change_type.value} for {self.target_object}"


@dataclass
class BreakingChange:
    """
    Represents a breaking change that could affect applications.

    Breaking changes are modifications that require application code changes
    or could cause runtime errors if not handled properly.
    """

    change_type: str  # 'column_removal', 'type_change', 'constraint_change', etc.
    description: str
    severity: ChangeImpact
    migration_strategy: str
    rollback_possibility: bool
    affected_queries: list[str] = field(default_factory=list)
    estimated_downtime: str | None = None
    risk_mitigation: list[str] = field(default_factory=list)


@dataclass
class OpsGuide:
    """
    Operational guide for deploying schema changes.

    This provides step-by-step instructions for safe deployment,
    including pre-deployment checks, deployment steps, and rollback procedures.
    """

    pre_deployment: list[str] = field(default_factory=list)
    deployment_steps: list[str] = field(default_factory=list)
    post_deployment: list[str] = field(default_factory=list)
    rollback_procedure: list[str] = field(default_factory=list)
    verification_steps: list[str] = field(default_factory=list)
    estimated_duration: str | None = None
    required_permissions: list[str] = field(default_factory=list)


@dataclass
class SchemaAnalysisInput:
    """Input data for schema analysis."""

    diff_content: str  # Unified diff content
    schema_files: list[str] = field(default_factory=list)  # Paths to schema files
    database_type: str = "postgresql"  # postgresql, mysql, sqlite, etc.
    include_ops_guide: bool = True
    include_breaking_analysis: bool = True
    custom_rules: dict[str, Any] | None = None


@dataclass
class SchemaAnalysisOutput:
    """Output from schema analysis."""

    ddl_changes: list[DDLChange] = field(default_factory=list)
    breaking_changes: list[BreakingChange] = field(default_factory=list)
    ops_guide: OpsGuide | None = None
    migration_complexity: MigrationComplexity = MigrationComplexity.LOW
    total_impact_score: float = 0.0
    risk_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)


class SchemaAnalyzer(BaseTool[SchemaAnalysisInput, SchemaAnalysisOutput]):
    """
    Comprehensive tool for analyzing database schema changes.

    This class provides methods to:
    1. Parse and analyze DDL statements from diff content
    2. Detect breaking changes that could affect applications
    3. Generate operational guides for safe deployments
    4. Assess migration complexity and provide risk analysis
    5. Generate comprehensive reports for stakeholders

    The tool supports multiple database types and can be customized
    with specific rules for different environments.
    """

    def __init__(self) -> None:
        """Initialize the Schema Analyzer."""
        super().__init__("SchemaAnalyzer")

        # DDL pattern matching for different database types
        self.ddl_patterns = {
            "postgresql": {
                "create_table": r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
                "create_index": r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s]+)",
                "alter_table": r"ALTER\s+TABLE\s+([^\s]+)\s+(ADD|DROP|ALTER|RENAME)",
                "drop_table": r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^\s;]+)",
                "add_column": r"ADD\s+COLUMN\s+([^\s]+)",
                "drop_column": r"DROP\s+COLUMN\s+(?:IF\s+EXISTS\s+)?([^\s;]+)",
                "alter_column": r"ALTER\s+COLUMN\s+([^\s]+)",
                "add_constraint": r"ADD\s+(CONSTRAINT\s+)?([^\s]+)",
                "drop_constraint": r"DROP\s+(CONSTRAINT\s+)?([^\s;]+)",
            },
            "mysql": {
                "create_table": r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
                "alter_table": r"ALTER\s+TABLE\s+([^\s]+)\s+(ADD|DROP|CHANGE|MODIFY)",
                "drop_table": r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^\s;]+)",
                "add_column": r"ADD\s+COLUMN\s+([^\s]+)",
                "drop_column": r"DROP\s+COLUMN\s+([^\s;]+)",
                "modify_column": r"MODIFY\s+COLUMN\s+([^\s]+)",
            },
            "sqlite": {
                "create_table": r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
                "alter_table": r"ALTER\s+TABLE\s+([^\s]+)\s+(ADD|DROP|RENAME)",
                "drop_table": r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?([^\s;]+)",
                "add_column": r"ADD\s+COLUMN\s+([^\s]+)",
            },
        }

        # Breaking change detection rules
        self.breaking_rules = {
            "column_removal": {
                "pattern": r"DROP\s+COLUMN",
                "severity": ChangeImpact.HIGH,
                "description": "Removing columns can break existing queries and application code",
            },
            "table_removal": {
                "pattern": r"DROP\s+TABLE",
                "severity": ChangeImpact.CRITICAL,
                "description": "Removing tables will break all dependent queries and applications",
            },
            "constraint_addition": {
                "pattern": r"ADD\s+(?:CONSTRAINT\s+)?(?:NOT\s+NULL|UNIQUE|PRIMARY\s+KEY)",
                "severity": ChangeImpact.MEDIUM,
                "description": "Adding constraints may fail if existing data violates them",
            },
            "type_change": {
                "pattern": r"ALTER\s+COLUMN.*TYPE",
                "severity": ChangeImpact.HIGH,
                "description": "Changing column types may cause data loss or application errors",
            },
        }

    def execute(
        self, input_data: SchemaAnalysisInput
    ) -> ToolResult[SchemaAnalysisOutput]:
        """
        Execute schema analysis on the provided input.

        Args:
            input_data: Input data containing diff content and analysis options

        Returns:
            ToolResult containing the analysis output or error information
        """
        try:
            # Validate input data
            if not self.validate_input(input_data):
                return ToolResult.error(
                    error_code=ToolErrorCode.INVALID_INPUT,
                    error_message="Invalid input data provided",
                    metrics=self._create_metrics(),
                )

            # Perform schema analysis
            output = self._perform_analysis(input_data)

            # Generate evidence
            evidence = self._create_evidence(input_data, output)

            # Create metrics
            metrics = self._create_metrics(
                files_processed=len(input_data.schema_files),
                lines_processed=len(input_data.diff_content.splitlines()),
            )

            return ToolResult.success(
                output=output,
                evidence=evidence,
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Schema analysis execution failed: {e}")
            return ToolResult.error(
                error_code=ToolErrorCode.PROCESSING_ERROR,
                error_message=str(e),
                metrics=self._create_metrics(),
            )

    def validate_input(self, input_data: SchemaAnalysisInput) -> bool:
        """
        Validate the input data for schema analysis.

        Args:
            input_data: Input data to validate

        Returns:
            True if input is valid, False otherwise
        """
        if not input_data.diff_content:
            logger.warning("No diff content provided for analysis")
            return False

        if input_data.database_type not in self.ddl_patterns:
            logger.warning(f"Unsupported database type: {input_data.database_type}")
            return False

        return True

    def _perform_analysis(
        self, input_data: SchemaAnalysisInput
    ) -> SchemaAnalysisOutput:
        """
        Perform the actual schema analysis.

        Args:
            input_data: Validated input data

        Returns:
            Complete analysis output
        """
        output = SchemaAnalysisOutput()

        # Parse DDL changes from diff content
        output.ddl_changes = self._parse_ddl_changes(
            input_data.diff_content, input_data.database_type
        )

        # Analyze breaking changes if requested
        if input_data.include_breaking_analysis:
            output.breaking_changes = self._analyze_breaking_changes(output.ddl_changes)

        # Generate operational guide if requested
        if input_data.include_ops_guide:
            output.ops_guide = self._generate_ops_guide(
                output.ddl_changes, output.breaking_changes
            )

        # Calculate migration complexity and impact
        output.migration_complexity = self._assess_migration_complexity(
            output.ddl_changes
        )
        output.total_impact_score = self._calculate_impact_score(
            output.ddl_changes, output.breaking_changes
        )

        # Generate risk summary and recommendations
        output.risk_summary = self._generate_risk_summary(
            output.ddl_changes, output.breaking_changes
        )
        output.recommendations = self._generate_recommendations(
            output.ddl_changes, output.breaking_changes, output.migration_complexity
        )

        return output

    def _parse_ddl_changes(
        self, diff_content: str, database_type: str
    ) -> list[DDLChange]:
        """
        Parse DDL changes from diff content.

        Args:
            diff_content: Unified diff content
            database_type: Type of database (postgresql, mysql, sqlite)

        Returns:
            List of detected DDL changes
        """
        changes = []
        patterns = self.ddl_patterns.get(database_type, {})

        # Split diff into lines and analyze each
        lines = diff_content.splitlines()

        for line_num, line in enumerate(lines, 1):
            # Look for DDL statements in added lines
            if line.startswith("+") and any(
                keyword in line.upper()
                for keyword in ["CREATE", "ALTER", "DROP", "ADD", "MODIFY"]
            ):
                change = self._parse_ddl_line(line[1:], patterns, line_num)
                if change:
                    changes.append(change)

        logger.info(f"Parsed {len(changes)} DDL changes from diff content")
        return changes

    def _parse_ddl_line(
        self, line: str, patterns: dict[str, str], line_number: int
    ) -> DDLChange | None:
        """
        Parse a single DDL line to extract change information.

        Args:
            line: DDL statement line
            patterns: Database-specific DDL patterns
            line_number: Line number in the diff

        Returns:
            DDLChange object if parsing successful, None otherwise
        """
        line_upper = line.upper().strip()

        # Determine change type and target
        if "CREATE TABLE" in line_upper:
            match = re.search(patterns.get("create_table", ""), line_upper)
            if match:
                return DDLChange(
                    change_type=ChangeType.CREATE,
                    target_object=match.group(1),
                    sql_statement=line,
                    impact=ChangeImpact.LOW,  # Creating tables is generally safe
                    line_number=line_number,
                )

        elif "CREATE INDEX" in line_upper:
            match = re.search(patterns.get("create_index", ""), line_upper)
            if match:
                return DDLChange(
                    change_type=ChangeType.INDEX,
                    target_object=match.group(1),
                    sql_statement=line,
                    impact=ChangeImpact.LOW,  # Creating indexes is generally safe
                    line_number=line_number,
                )

        elif "ALTER TABLE" in line_upper:
            # Check for different types of ALTER operations
            if "ADD COLUMN" in line_upper:
                match = re.search(patterns.get("add_column", ""), line_upper)
                if match:
                    return DDLChange(
                        change_type=ChangeType.ALTER,
                        target_object=match.group(1),
                        sql_statement=line,
                        impact=ChangeImpact.LOW,  # Adding columns is generally safe
                        line_number=line_number,
                    )

            elif "DROP COLUMN" in line_upper:
                match = re.search(patterns.get("drop_column", ""), line_upper)
                if match:
                    return DDLChange(
                        change_type=ChangeType.ALTER,
                        target_object=match.group(1),
                        sql_statement=line,
                        impact=ChangeImpact.HIGH,  # Dropping columns can break applications
                        line_number=line_number,
                    )

            elif "ADD CONSTRAINT" in line_upper:
                match = re.search(patterns.get("add_constraint", ""), line_upper)
                if match:
                    return DDLChange(
                        change_type=ChangeType.CONSTRAINT,
                        target_object=match.group(2)
                        if match.group(1) == "CONSTRAINT"
                        else match.group(1),
                        sql_statement=line,
                        impact=ChangeImpact.MEDIUM,  # Adding constraints may fail
                        line_number=line_number,
                    )

        elif "DROP TABLE" in line_upper:
            match = re.search(patterns.get("drop_table", ""), line_upper)
            if match:
                return DDLChange(
                    change_type=ChangeType.DROP,
                    target_object=match.group(1),
                    sql_statement=line,
                    impact=ChangeImpact.CRITICAL,  # Dropping tables is very dangerous
                    line_number=line_number,
                )

        return None

    def _analyze_breaking_changes(
        self, ddl_changes: list[DDLChange]
    ) -> list[BreakingChange]:
        """
        Analyze DDL changes to identify breaking changes.

        Args:
            ddl_changes: List of parsed DDL changes

        Returns:
            List of identified breaking changes
        """
        breaking_changes = []

        for change in ddl_changes:
            breaking_change = self._assess_breaking_change(change)
            if breaking_change:
                breaking_changes.append(breaking_change)

        logger.info(f"Identified {len(breaking_changes)} breaking changes")
        return breaking_changes

    def _assess_breaking_change(self, change: DDLChange) -> BreakingChange | None:
        """
        Assess whether a single DDL change is a breaking change.

        Args:
            change: DDL change to assess

        Returns:
            BreakingChange object if breaking, None otherwise
        """
        # High impact changes are likely breaking
        if change.impact in [ChangeImpact.HIGH, ChangeImpact.CRITICAL]:
            if change.change_type == ChangeType.DROP:
                return BreakingChange(
                    change_type="object_removal",
                    description=f"Removing {change.target_object} will break dependent code",
                    severity=change.impact,
                    migration_strategy="Coordinate with development team to update application code",
                    rollback_possibility=False,  # Cannot rollback DROP operations
                    risk_mitigation=[
                        "Ensure no application code references the removed object",
                        "Update all dependent queries and application logic",
                        "Test thoroughly in staging environment",
                    ],
                )

            elif change.change_type == ChangeType.ALTER:
                if "DROP COLUMN" in change.sql_statement.upper():
                    return BreakingChange(
                        change_type="column_removal",
                        description=f"Removing column {change.target_object} will break queries",
                        severity=change.impact,
                        migration_strategy="Update application code to remove column references",
                        rollback_possibility=False,
                        risk_mitigation=[
                            "Identify all queries using the removed column",
                            "Update application code and queries",
                            "Verify data integrity after changes",
                        ],
                    )

        return None

    def _generate_ops_guide(
        self, ddl_changes: list[DDLChange], breaking_changes: list[BreakingChange]
    ) -> OpsGuide:
        """
        Generate operational guide for deploying schema changes.

        Args:
            ddl_changes: List of DDL changes
            breaking_changes: List of breaking changes

        Returns:
            Complete operational guide
        """
        guide = OpsGuide()

        # Pre-deployment steps
        guide.pre_deployment = [
            "Review all DDL changes with development team",
            "Ensure backup of current database schema and data",
            "Verify application compatibility with proposed changes",
            "Schedule maintenance window if breaking changes exist",
            "Prepare rollback scripts for all changes",
        ]

        # Deployment steps
        guide.deployment_steps = [
            "Stop application services gracefully",
            "Execute DDL changes in order of dependencies",
            "Verify schema changes were applied correctly",
            "Update application configuration if needed",
            "Restart application services",
        ]

        # Post-deployment steps
        guide.post_deployment = [
            "Verify application functionality",
            "Check database performance metrics",
            "Monitor error logs for any issues",
            "Validate data integrity",
            "Update documentation with new schema",
        ]

        # Rollback procedure
        if breaking_changes:
            guide.rollback_procedure = [
                "Stop application services immediately",
                "Restore database from backup",
                "Revert any application configuration changes",
                "Verify system is back to previous state",
                "Investigate root cause of deployment failure",
            ]
        else:
            guide.rollback_procedure = [
                "Execute reverse DDL statements if possible",
                "Restore from backup if needed",
                "Verify system stability after rollback",
            ]

        # Verification steps
        guide.verification_steps = [
            "Run application smoke tests",
            "Verify database connectivity",
            "Check application logs for errors",
            "Validate critical business functions",
            "Monitor system performance",
        ]

        guide.estimated_duration = "15-30 minutes"
        guide.required_permissions = ["DB_OWNER", "CREATE", "ALTER", "DROP"]

        return guide

    def _assess_migration_complexity(
        self, ddl_changes: list[DDLChange]
    ) -> MigrationComplexity:
        """
        Assess the complexity of the migration based on DDL changes.

        Args:
            ddl_changes: List of DDL changes

        Returns:
            Migration complexity level
        """
        if not ddl_changes:
            return MigrationComplexity.LOW

        # Count high-impact changes
        high_impact_count = sum(
            1
            for change in ddl_changes
            if change.impact in [ChangeImpact.HIGH, ChangeImpact.CRITICAL]
        )

        # Count destructive operations
        destructive_count = sum(
            1
            for change in ddl_changes
            if change.change_type in [ChangeType.DROP, ChangeType.ALTER]
        )

        if high_impact_count >= 3 or destructive_count >= 2:
            return MigrationComplexity.HIGH
        elif high_impact_count >= 1 or destructive_count >= 1:
            return MigrationComplexity.MEDIUM
        else:
            return MigrationComplexity.LOW

    def _calculate_impact_score(
        self, ddl_changes: list[DDLChange], breaking_changes: list[BreakingChange]
    ) -> float:
        """
        Calculate overall impact score for the schema changes.

        Args:
            ddl_changes: List of DDL changes
            breaking_changes: List of breaking changes

        Returns:
            Impact score from 0.0 to 10.0
        """
        if not ddl_changes:
            return 0.0

        # Base score from DDL changes
        base_score = sum(
            {"low": 1, "medium": 3, "high": 7, "critical": 10}[change.impact.value]
            for change in ddl_changes
        )

        # Additional score from breaking changes
        breaking_score = sum(
            {"low": 2, "medium": 5, "high": 8, "critical": 10}[change.severity.value]
            for change in breaking_changes
        )

        # Normalize to 0-10 scale
        total_score = (base_score + breaking_score) / len(ddl_changes)
        return min(total_score, 10.0)

    def _generate_risk_summary(
        self, ddl_changes: list[DDLChange], breaking_changes: list[BreakingChange]
    ) -> dict[str, int | str]:
        """
        Generate a summary of risks associated with the schema changes.

        Args:
            ddl_changes: List of DDL changes
            breaking_changes: List of breaking changes

        Returns:
            Risk summary dictionary with typed values
        """
        total_changes = len(ddl_changes)
        breaking_changes_count = len(breaking_changes)
        high_risk_changes = len(
            [
                c
                for c in ddl_changes
                if c.impact in [ChangeImpact.HIGH, ChangeImpact.CRITICAL]
            ]
        )
        destructive_operations = len(
            [c for c in ddl_changes if c.change_type == ChangeType.DROP]
        )

        # Determine overall risk level
        if breaking_changes_count > 0 or high_risk_changes > 2:
            risk_level = "high"
        elif high_risk_changes > 0 or destructive_operations > 0:
            risk_level = "medium"
        else:
            risk_level = "low"

        risk_summary: dict[str, int | str] = {
            "total_changes": total_changes,
            "breaking_changes": breaking_changes_count,
            "high_risk_changes": high_risk_changes,
            "destructive_operations": destructive_operations,
            "risk_level": risk_level,
        }

        return risk_summary

    def _generate_recommendations(
        self,
        ddl_changes: list[DDLChange],
        breaking_changes: list[BreakingChange],
        complexity: MigrationComplexity,
    ) -> list[str]:
        """
        Generate recommendations for safe deployment.

        Args:
            ddl_changes: List of DDL changes
            breaking_changes: List of breaking changes
            complexity: Migration complexity level

        Returns:
            List of recommendations
        """
        recommendations = []

        if breaking_changes:
            recommendations.append(
                "Schedule maintenance window for deployment due to breaking changes"
            )
            recommendations.append(
                "Coordinate closely with development team for application updates"
            )

        if complexity == MigrationComplexity.HIGH:
            recommendations.append(
                "Consider breaking migration into smaller, safer phases"
            )
            recommendations.append("Perform thorough testing in staging environment")

        if any(c.change_type == ChangeType.DROP for c in ddl_changes):
            recommendations.append(
                "Ensure comprehensive backup before destructive operations"
            )
            recommendations.append(
                "Verify no critical dependencies exist on objects being removed"
            )

        if not recommendations:
            recommendations.append(
                "Changes appear safe for deployment during business hours"
            )

        return recommendations

    def _create_evidence(
        self, input_data: SchemaAnalysisInput, output: SchemaAnalysisOutput
    ) -> list[ToolEvidence]:
        """
        Create evidence for the analysis results.

        Args:
            input_data: Input data used for analysis
            output: Analysis output

        Returns:
            List of evidence items
        """
        evidence = []

        # Evidence for DDL changes
        if output.ddl_changes:
            evidence.append(
                ToolEvidence(
                    file_path="schema_analysis",
                    content=f"Analyzed {len(output.ddl_changes)} DDL changes",
                    evidence_type="ddl_analysis",
                    description="Number of DDL changes detected in diff content",
                )
            )

        # Evidence for breaking changes
        if output.breaking_changes:
            evidence.append(
                ToolEvidence(
                    file_path="schema_analysis",
                    content=f"Identified {len(output.breaking_changes)} breaking changes",
                    evidence_type="breaking_changes",
                    description="Number of breaking changes that require attention",
                )
            )

        # Evidence for migration complexity
        evidence.append(
            ToolEvidence(
                file_path="schema_analysis",
                content=f"Migration complexity: {output.migration_complexity.value}",
                evidence_type="complexity_assessment",
                description="Assessed complexity level for the migration",
            )
        )

        return evidence


def main() -> None:
    """
    Example usage of SchemaAnalyzer class.

    This demonstrates how to use the SchemaAnalyzer to:
    1. Analyze DDL changes from diff content
    2. Detect breaking changes
    3. Generate operational guides
    4. Assess migration complexity and risks
    """
    import sys

    # Example diff content (in real usage, this would come from git diff)
    example_diff = """
+ CREATE TABLE users (
+   id SERIAL PRIMARY KEY,
+   username VARCHAR(50) UNIQUE NOT NULL,
+   email VARCHAR(100) UNIQUE NOT NULL
+ );
+
+ ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NOW();
+
+ CREATE INDEX idx_users_email ON users(email);
    """

    try:
        # Initialize the analyzer
        analyzer = SchemaAnalyzer()

        # Create input data
        input_data = SchemaAnalysisInput(
            diff_content=example_diff,
            database_type="postgresql",
            include_ops_guide=True,
            include_breaking_analysis=True,
        )

        # Run analysis
        logger.info("Running schema analysis...")
        result = analyzer.run(input_data)

        if result.status.value == "success" and result.output:
            output = result.output
            logger.info("Analysis completed successfully!")
            logger.info(f"DDL Changes: {len(output.ddl_changes)}")
            logger.info(f"Breaking Changes: {len(output.breaking_changes)}")
            logger.info(f"Migration Complexity: {output.migration_complexity.value}")
            logger.info(f"Impact Score: {output.total_impact_score:.2f}")

            # Display recommendations
            if output.recommendations:
                logger.info("\nRecommendations:")
                for rec in output.recommendations:
                    logger.info(f"  • {rec}")
        else:
            logger.error(f"Analysis failed: {result.error_message}")

    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
