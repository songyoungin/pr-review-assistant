"""
Tests for SchemaAnalyzer tool.

This module contains comprehensive tests for the SchemaAnalyzer tool,
covering DDL parsing, breaking change detection, and operational guide generation.
"""

from unittest.mock import MagicMock, patch

import pytest

from tools.schema_analysis.schema_analyzer import (
    BreakingChange,
    ChangeImpact,
    ChangeType,
    DDLChange,
    MigrationComplexity,
    OpsGuide,
    SchemaAnalysisInput,
    SchemaAnalysisOutput,
    SchemaAnalyzer,
)


class TestDDLChange:
    """Test DDLChange dataclass functionality."""

    def test_ddl_change_creation(self) -> None:
        """Test creating a DDLChange instance."""
        change = DDLChange(
            change_type=ChangeType.CREATE,
            target_object="users",
            sql_statement="CREATE TABLE users (id SERIAL PRIMARY KEY);",
            impact=ChangeImpact.LOW,
        )

        assert change.change_type == ChangeType.CREATE
        assert change.target_object == "users"
        assert change.impact == ChangeImpact.LOW
        assert change.description == "Create new users"

    def test_ddl_change_description_generation(self) -> None:
        """Test automatic description generation for different change types."""
        # Test CREATE
        create_change = DDLChange(
            change_type=ChangeType.CREATE,
            target_object="products",
            sql_statement="CREATE TABLE products;",
            impact=ChangeImpact.LOW,
        )
        assert create_change.description == "Create new products"

        # Test ALTER
        alter_change = DDLChange(
            change_type=ChangeType.ALTER,
            target_object="users",
            sql_statement="ALTER TABLE users ADD COLUMN email;",
            impact=ChangeImpact.MEDIUM,
        )
        assert alter_change.description == "Modify existing users"

        # Test DROP
        drop_change = DDLChange(
            change_type=ChangeType.DROP,
            target_object="temp_table",
            sql_statement="DROP TABLE temp_table;",
            impact=ChangeImpact.HIGH,
        )
        assert drop_change.description == "Remove temp_table"


class TestBreakingChange:
    """Test BreakingChange dataclass functionality."""

    def test_breaking_change_creation(self) -> None:
        """Test creating a BreakingChange instance."""
        breaking_change = BreakingChange(
            change_type="column_removal",
            description="Removing email column will break queries",
            severity=ChangeImpact.HIGH,
            migration_strategy="Update application code",
            rollback_possibility=False,
        )

        assert breaking_change.change_type == "column_removal"
        assert breaking_change.severity == ChangeImpact.HIGH
        assert breaking_change.rollback_possibility is False


class TestOpsGuide:
    """Test OpsGuide dataclass functionality."""

    def test_ops_guide_creation(self) -> None:
        """Test creating an OpsGuide instance."""
        guide = OpsGuide()

        assert isinstance(guide.pre_deployment, list)
        assert isinstance(guide.deployment_steps, list)
        assert isinstance(guide.post_deployment, list)
        assert isinstance(guide.rollback_procedure, list)


class TestSchemaAnalysisInput:
    """Test SchemaAnalysisInput dataclass functionality."""

    def test_schema_analysis_input_creation(self) -> None:
        """Test creating a SchemaAnalysisInput instance."""
        input_data = SchemaAnalysisInput(
            diff_content="+ CREATE TABLE users;",
            database_type="postgresql",
            include_ops_guide=True,
            include_breaking_analysis=True,
        )

        assert input_data.diff_content == "+ CREATE TABLE users;"
        assert input_data.database_type == "postgresql"
        assert input_data.include_ops_guide is True
        assert input_data.include_breaking_analysis is True

    def test_schema_analysis_input_defaults(self) -> None:
        """Test SchemaAnalysisInput default values."""
        input_data = SchemaAnalysisInput(diff_content="test")

        assert input_data.database_type == "postgresql"
        assert input_data.include_ops_guide is True
        assert input_data.include_breaking_analysis is True
        assert input_data.schema_files == []


class TestSchemaAnalysisOutput:
    """Test SchemaAnalysisOutput dataclass functionality."""

    def test_schema_analysis_output_creation(self) -> None:
        """Test creating a SchemaAnalysisOutput instance."""
        output = SchemaAnalysisOutput()

        assert output.ddl_changes == []
        assert output.breaking_changes == []
        assert output.migration_complexity == MigrationComplexity.LOW
        assert output.total_impact_score == 0.0


class TestSchemaAnalyzer:
    """Test SchemaAnalyzer tool functionality."""

    @pytest.fixture
    def analyzer(self) -> SchemaAnalyzer:
        """Create a SchemaAnalyzer instance for testing."""
        return SchemaAnalyzer()

    @pytest.fixture
    def sample_diff_content(self) -> str:
        """Sample diff content for testing."""
        return """
+ CREATE TABLE users (
+   id SERIAL PRIMARY KEY,
+   username VARCHAR(50) UNIQUE NOT NULL,
+   email VARCHAR(100) UNIQUE NOT NULL
+ );

+ ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NOW();

+ CREATE INDEX idx_users_email ON users(email);
        """

    def test_analyzer_initialization(self, analyzer: SchemaAnalyzer) -> None:
        """Test SchemaAnalyzer initialization."""
        assert analyzer.tool_name == "SchemaAnalyzer"
        assert "postgresql" in analyzer.ddl_patterns
        assert "mysql" in analyzer.ddl_patterns
        assert "sqlite" in analyzer.ddl_patterns

    def test_validate_input_valid(self, analyzer: SchemaAnalyzer) -> None:
        """Test input validation with valid data."""
        input_data = SchemaAnalysisInput(
            diff_content="+ CREATE TABLE test;", database_type="postgresql"
        )

        assert analyzer.validate_input(input_data) is True

    def test_validate_input_empty_diff(self, analyzer: SchemaAnalyzer) -> None:
        """Test input validation with empty diff content."""
        input_data = SchemaAnalysisInput(diff_content="", database_type="postgresql")

        assert analyzer.validate_input(input_data) is False

    def test_validate_input_unsupported_database(
        self, analyzer: SchemaAnalyzer
    ) -> None:
        """Test input validation with unsupported database type."""
        input_data = SchemaAnalysisInput(
            diff_content="+ CREATE TABLE test;", database_type="unsupported_db"
        )

        assert analyzer.validate_input(input_data) is False

    def test_parse_ddl_changes_postgresql(
        self, analyzer: SchemaAnalyzer, sample_diff_content: str
    ) -> None:
        """Test DDL parsing for PostgreSQL."""
        changes = analyzer._parse_ddl_changes(sample_diff_content, "postgresql")

        assert len(changes) == 3

        # Check CREATE TABLE
        create_change = next(c for c in changes if c.change_type == ChangeType.CREATE)
        assert create_change.target_object == "USERS"  # Regex captures uppercase
        assert create_change.impact == ChangeImpact.LOW

        # Check ALTER TABLE
        alter_change = next(c for c in changes if "ADD COLUMN" in c.sql_statement)
        assert alter_change.change_type == ChangeType.ALTER
        assert alter_change.impact == ChangeImpact.LOW

        # Check CREATE INDEX
        index_change = next(c for c in changes if c.change_type == ChangeType.INDEX)
        assert index_change.target_object == "IDX_USERS_EMAIL"
        assert index_change.impact == ChangeImpact.LOW

    def test_parse_ddl_changes_mysql(self, analyzer: SchemaAnalyzer) -> None:
        """Test DDL parsing for MySQL."""
        mysql_diff = """
+ CREATE TABLE products (
+   id INT AUTO_INCREMENT PRIMARY KEY,
+   name VARCHAR(100) NOT NULL
+ );
+
+ ALTER TABLE products MODIFY COLUMN name VARCHAR(200);
        """

        changes = analyzer._parse_ddl_changes(mysql_diff, "mysql")
        assert len(changes) >= 1

    def test_parse_ddl_changes_sqlite(self, analyzer: SchemaAnalyzer) -> None:
        """Test DDL parsing for SQLite."""
        sqlite_diff = """
+ CREATE TABLE logs (
+   id INTEGER PRIMARY KEY,
+   message TEXT
+ );
        """

        changes = analyzer._parse_ddl_changes(sqlite_diff, "sqlite")
        assert len(changes) >= 1

    def test_parse_ddl_line_create_table(self, analyzer: SchemaAnalyzer) -> None:
        """Test parsing CREATE TABLE statements."""
        line = "CREATE TABLE users (id SERIAL PRIMARY KEY);"
        change = analyzer._parse_ddl_line(line, analyzer.ddl_patterns["postgresql"], 1)

        assert change is not None
        assert change.change_type == ChangeType.CREATE
        assert change.target_object == "USERS"  # Regex captures uppercase
        assert change.impact == ChangeImpact.LOW

    def test_parse_ddl_line_alter_table_add_column(
        self, analyzer: SchemaAnalyzer
    ) -> None:
        """Test parsing ALTER TABLE ADD COLUMN statements."""
        line = "ALTER TABLE users ADD COLUMN email VARCHAR(100);"
        change = analyzer._parse_ddl_line(line, analyzer.ddl_patterns["postgresql"], 1)

        assert change is not None
        assert change.change_type == ChangeType.ALTER
        assert change.target_object == "EMAIL"  # Regex captures uppercase
        assert change.impact == ChangeImpact.LOW

    def test_parse_ddl_line_drop_table(self, analyzer: SchemaAnalyzer) -> None:
        """Test parsing DROP TABLE statements."""
        line = "DROP TABLE temp_table;"
        change = analyzer._parse_ddl_line(line, analyzer.ddl_patterns["postgresql"], 1)

        assert change is not None
        assert change.change_type == ChangeType.DROP
        assert change.target_object == "TEMP_TABLE"  # Regex captures uppercase
        assert change.impact == ChangeImpact.CRITICAL

    def test_parse_ddl_line_unsupported(self, analyzer: SchemaAnalyzer) -> None:
        """Test parsing unsupported DDL statements."""
        line = "SELECT * FROM users;"
        change = analyzer._parse_ddl_line(line, analyzer.ddl_patterns["postgresql"], 1)

        assert change is None

    def test_analyze_breaking_changes(self, analyzer: SchemaAnalyzer) -> None:
        """Test breaking change analysis."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="users",
                sql_statement="DROP TABLE users;",
                impact=ChangeImpact.CRITICAL,
            ),
            DDLChange(
                change_type=ChangeType.ALTER,
                target_object="email",
                sql_statement="ALTER TABLE users DROP COLUMN email;",
                impact=ChangeImpact.HIGH,
            ),
        ]

        breaking_changes = analyzer._analyze_breaking_changes(ddl_changes)

        assert len(breaking_changes) == 2

        # Check table removal breaking change
        table_removal = next(
            c for c in breaking_changes if c.change_type == "object_removal"
        )
        assert table_removal.severity == ChangeImpact.CRITICAL
        assert table_removal.rollback_possibility is False

        # Check column removal breaking change
        column_removal = next(
            c for c in breaking_changes if c.change_type == "column_removal"
        )
        assert column_removal.severity == ChangeImpact.HIGH
        assert column_removal.rollback_possibility is False

    def test_assess_breaking_change_high_impact(self, analyzer: SchemaAnalyzer) -> None:
        """Test breaking change assessment for high-impact changes."""
        change = DDLChange(
            change_type=ChangeType.DROP,
            target_object="users",
            sql_statement="DROP TABLE users;",
            impact=ChangeImpact.CRITICAL,
        )

        breaking_change = analyzer._assess_breaking_change(change)

        assert breaking_change is not None
        assert breaking_change.change_type == "object_removal"
        assert breaking_change.severity == ChangeImpact.CRITICAL
        assert breaking_change.rollback_possibility is False

    def test_assess_breaking_change_low_impact(self, analyzer: SchemaAnalyzer) -> None:
        """Test breaking change assessment for low-impact changes."""
        change = DDLChange(
            change_type=ChangeType.CREATE,
            target_object="users",
            sql_statement="CREATE TABLE users;",
            impact=ChangeImpact.LOW,
        )

        breaking_change = analyzer._assess_breaking_change(change)

        assert breaking_change is None

    def test_generate_ops_guide(self, analyzer: SchemaAnalyzer) -> None:
        """Test operational guide generation."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.CREATE,
                target_object="users",
                sql_statement="CREATE TABLE users;",
                impact=ChangeImpact.LOW,
            )
        ]

        breaking_changes: list[BreakingChange] = []

        guide = analyzer._generate_ops_guide(ddl_changes, breaking_changes)

        assert isinstance(guide, OpsGuide)
        assert len(guide.pre_deployment) > 0
        assert len(guide.deployment_steps) > 0
        assert len(guide.post_deployment) > 0
        assert len(guide.rollback_procedure) > 0
        assert len(guide.verification_steps) > 0

    def test_generate_ops_guide_with_breaking_changes(
        self, analyzer: SchemaAnalyzer
    ) -> None:
        """Test operational guide generation with breaking changes."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="users",
                sql_statement="DROP TABLE users;",
                impact=ChangeImpact.CRITICAL,
            )
        ]

        breaking_changes = [
            BreakingChange(
                change_type="object_removal",
                description="Removing users table will break applications",
                severity=ChangeImpact.CRITICAL,
                migration_strategy="Coordinate with development team",
                rollback_possibility=False,
            )
        ]

        guide = analyzer._generate_ops_guide(ddl_changes, breaking_changes)

        # Should have more aggressive rollback procedure
        assert any("immediately" in step.lower() for step in guide.rollback_procedure)

    def test_assess_migration_complexity_low(self, analyzer: SchemaAnalyzer) -> None:
        """Test migration complexity assessment for low complexity."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.CREATE,
                target_object="users",
                sql_statement="CREATE TABLE users;",
                impact=ChangeImpact.LOW,
            )
        ]

        complexity = analyzer._assess_migration_complexity(ddl_changes)

        assert complexity == MigrationComplexity.LOW

    def test_assess_migration_complexity_medium(self, analyzer: SchemaAnalyzer) -> None:
        """Test migration complexity assessment for medium complexity."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.ALTER,
                target_object="users",
                sql_statement="ALTER TABLE users DROP COLUMN email;",
                impact=ChangeImpact.HIGH,
            )
        ]

        complexity = analyzer._assess_migration_complexity(ddl_changes)

        assert complexity == MigrationComplexity.MEDIUM

    def test_assess_migration_complexity_high(self, analyzer: SchemaAnalyzer) -> None:
        """Test migration complexity assessment for high complexity."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="users",
                sql_statement="DROP TABLE users;",
                impact=ChangeImpact.CRITICAL,
            ),
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="products",
                sql_statement="DROP TABLE products;",
                impact=ChangeImpact.CRITICAL,
            ),
            DDLChange(
                change_type=ChangeType.ALTER,
                target_object="orders",
                sql_statement="ALTER TABLE orders DROP COLUMN status;",
                impact=ChangeImpact.HIGH,
            ),
        ]

        complexity = analyzer._assess_migration_complexity(ddl_changes)

        assert complexity == MigrationComplexity.HIGH

    def test_calculate_impact_score(self, analyzer: SchemaAnalyzer) -> None:
        """Test impact score calculation."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.CREATE,
                target_object="users",
                sql_statement="CREATE TABLE users;",
                impact=ChangeImpact.LOW,
            ),
            DDLChange(
                change_type=ChangeType.ALTER,
                target_object="users",
                sql_statement="ALTER TABLE users DROP COLUMN email;",
                impact=ChangeImpact.HIGH,
            ),
        ]

        breaking_changes = [
            BreakingChange(
                change_type="column_removal",
                description="Removing email column will break queries",
                severity=ChangeImpact.HIGH,
                migration_strategy="Update application code",
                rollback_possibility=False,
            )
        ]

        score = analyzer._calculate_impact_score(ddl_changes, breaking_changes)

        assert 0.0 <= score <= 10.0
        assert score > 0.0

    def test_generate_risk_summary(self, analyzer: SchemaAnalyzer) -> None:
        """Test risk summary generation."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="users",
                sql_statement="DROP TABLE users;",
                impact=ChangeImpact.CRITICAL,
            )
        ]

        breaking_changes = [
            BreakingChange(
                change_type="object_removal",
                description="Removing users table will break applications",
                severity=ChangeImpact.CRITICAL,
                migration_strategy="Coordinate with development team",
                rollback_possibility=False,
            )
        ]

        risk_summary = analyzer._generate_risk_summary(ddl_changes, breaking_changes)

        assert risk_summary["total_changes"] == 1
        assert risk_summary["breaking_changes"] == 1
        assert risk_summary["high_risk_changes"] == 1
        assert risk_summary["destructive_operations"] == 1
        assert risk_summary["risk_level"] == "high"

    def test_generate_recommendations(self, analyzer: SchemaAnalyzer) -> None:
        """Test recommendations generation."""
        ddl_changes = [
            DDLChange(
                change_type=ChangeType.DROP,
                target_object="users",
                sql_statement="DROP TABLE users;",
                impact=ChangeImpact.CRITICAL,
            )
        ]

        breaking_changes = [
            BreakingChange(
                change_type="object_removal",
                description="Removing users table will break applications",
                severity=ChangeImpact.CRITICAL,
                migration_strategy="Coordinate with development team",
                rollback_possibility=False,
            )
        ]

        complexity = MigrationComplexity.HIGH

        recommendations = analyzer._generate_recommendations(
            ddl_changes, breaking_changes, complexity
        )

        assert len(recommendations) > 0
        assert any("maintenance window" in rec.lower() for rec in recommendations)
        assert any("backup" in rec.lower() for rec in recommendations)

    def test_create_evidence(self, analyzer: SchemaAnalyzer) -> None:
        """Test evidence creation."""
        input_data = SchemaAnalysisInput(
            diff_content="+ CREATE TABLE users;", database_type="postgresql"
        )

        output = SchemaAnalysisOutput()
        output.ddl_changes = [
            DDLChange(
                change_type=ChangeType.CREATE,
                target_object="users",
                sql_statement="CREATE TABLE users;",
                impact=ChangeImpact.LOW,
            )
        ]
        output.migration_complexity = MigrationComplexity.LOW

        evidence = analyzer._create_evidence(input_data, output)

        assert len(evidence) >= 2  # DDL changes + complexity assessment
        assert any(e.evidence_type == "ddl_analysis" for e in evidence)
        assert any(e.evidence_type == "complexity_assessment" for e in evidence)

    @patch("tools.schema_analysis.schema_analyzer.logger")
    def test_execute_success(
        self, mock_logger: MagicMock, analyzer: SchemaAnalyzer, sample_diff_content: str
    ) -> None:
        """Test successful execution of schema analysis."""
        input_data = SchemaAnalysisInput(
            diff_content=sample_diff_content, database_type="postgresql"
        )

        result = analyzer.execute(input_data)

        assert result.status.value == "success"
        assert result.output is not None
        assert len(result.output.ddl_changes) > 0
        assert result.evidence is not None
        assert result.metrics is not None

    @patch("tools.schema_analysis.schema_analyzer.logger")
    def test_execute_invalid_input(
        self, mock_logger: MagicMock, analyzer: SchemaAnalyzer
    ) -> None:
        """Test execution with invalid input."""
        input_data = SchemaAnalysisInput(diff_content="", database_type="postgresql")

        result = analyzer.execute(input_data)

        assert result.status.value == "error"
        assert result.error_code is not None
        assert result.error_code.value == "INVALID_INPUT"
        assert result.output is None

    @patch("tools.schema_analysis.schema_analyzer.logger")
    def test_execute_processing_error(
        self, mock_logger: MagicMock, analyzer: SchemaAnalyzer
    ) -> None:
        """Test execution with processing error."""
        # Mock _perform_analysis to raise an exception
        with patch.object(
            analyzer, "_perform_analysis", side_effect=Exception("Test error")
        ):
            input_data = SchemaAnalysisInput(
                diff_content="+ CREATE TABLE users;", database_type="postgresql"
            )

            result = analyzer.execute(input_data)

            assert result.status.value == "error"
            assert result.error_code is not None
            assert result.error_code.value == "PROCESSING_ERROR"
            assert result.error_message is not None
            assert "Test error" in result.error_message


class TestSchemaAnalyzerIntegration:
    """Integration tests for SchemaAnalyzer."""

    @pytest.fixture
    def analyzer(self) -> SchemaAnalyzer:
        """Create a SchemaAnalyzer instance for integration testing."""
        return SchemaAnalyzer()

    def test_complete_analysis_workflow(self, analyzer: SchemaAnalyzer) -> None:
        """Test complete analysis workflow from input to output."""
        # Complex diff content with multiple change types
        diff_content = """
+ CREATE TABLE users (
+   id SERIAL PRIMARY KEY,
+   username VARCHAR(50) UNIQUE NOT NULL,
+   email VARCHAR(100) UNIQUE NOT NULL
+ );
+
+ ALTER TABLE users ADD COLUMN created_at TIMESTAMP DEFAULT NOW();
+
+ CREATE INDEX idx_users_email ON users(email);
+
+ DROP TABLE temp_logs;
+
+ ALTER TABLE products DROP COLUMN deprecated_flag;
        """

        input_data = SchemaAnalysisInput(
            diff_content=diff_content,
            database_type="postgresql",
            include_ops_guide=True,
            include_breaking_analysis=True,
        )

        # Execute analysis
        result = analyzer.execute(input_data)

        # Verify successful execution
        assert result.status.value == "success"
        assert result.output is not None

        output = result.output

        # Verify DDL changes were parsed
        assert len(output.ddl_changes) >= 4

        # Verify breaking changes were detected
        assert len(output.breaking_changes) >= 2  # DROP TABLE and DROP COLUMN

        # Verify operational guide was generated
        assert output.ops_guide is not None
        assert len(output.ops_guide.pre_deployment) > 0
        assert len(output.ops_guide.rollback_procedure) > 0

        # Verify complexity assessment
        assert output.migration_complexity in [
            MigrationComplexity.MEDIUM,
            MigrationComplexity.HIGH,
        ]

        # Verify impact score
        assert output.total_impact_score > 0.0

        # Verify risk summary
        assert output.risk_summary["risk_level"] in ["medium", "high"]

        # Verify recommendations
        assert len(output.recommendations) > 0

    def test_different_database_types(self, analyzer: SchemaAnalyzer) -> None:
        """Test analysis with different database types."""
        databases = ["postgresql", "mysql", "sqlite"]

        for db_type in databases:
            diff_content = f"+ CREATE TABLE test_{db_type} (id INT);"

            input_data = SchemaAnalysisInput(
                diff_content=diff_content, database_type=db_type
            )

            result = analyzer.execute(input_data)

            assert result.status.value == "success"
            assert result.output is not None
            assert len(result.output.ddl_changes) >= 1

    def test_edge_cases(self, analyzer: SchemaAnalyzer) -> None:
        """Test edge cases and boundary conditions."""
        # Empty diff content
        input_data = SchemaAnalysisInput(diff_content="", database_type="postgresql")

        result = analyzer.execute(input_data)
        assert result.status.value == "error"

        # Very long diff content
        long_diff = "+ CREATE TABLE test;\n" * 1000

        input_data = SchemaAnalysisInput(
            diff_content=long_diff, database_type="postgresql"
        )

        result = analyzer.execute(input_data)
        assert result.status.value == "success"
        assert result.output is not None
        assert len(result.output.ddl_changes) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
