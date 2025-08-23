"""
Rule management system for PR Review Assistant.

This module manages all rules used in the PR review system with their IDs and versions
for traceability (NFR-5 compliance). Each rule has a unique ID and version information
enabling change tracking.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class RuleCategory(Enum):
    """
    Enumeration of rule categories.

    Attributes:
        SECURITY: Security-related rules
        PERFORMANCE: Performance optimization rules
        QUALITY: Code quality rules
        STYLE: Code style rules
        DOCUMENTATION: Documentation rules
        SCHEMA: Database schema rules
    """

    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    STYLE = "style"
    DOCUMENTATION = "documentation"
    SCHEMA = "schema"


class Severity(Enum):
    """
    Enumeration of rule severity levels.

    Attributes:
        CRITICAL: Critical severity issues
        HIGH: High severity issues
        MEDIUM: Medium severity issues
        LOW: Low severity issues
        INFO: Informational issues
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class RuleDefinition:
    """
    Rule definition structure.

    Contains rule metadata for NFR-5 compliance.
    """

    rule_id: str  # Unique rule ID (e.g., "SEC001", "PERF001")
    name: str  # Rule name
    category: RuleCategory  # Rule category
    severity: Severity  # Default severity
    version: str  # Rule version (e.g., "1.0.0")
    description: str  # Rule description
    examples: list[str]  # Application examples
    created_at: str  # Creation date
    updated_at: str  # Last modified date
    deprecated: bool = False  # Deprecation status
    replacement_rule: str | None = None  # Replacement rule ID

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "category": self.category.value,
            "severity": self.severity.value,
            "version": self.version,
            "description": self.description,
            "examples": self.examples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "deprecated": self.deprecated,
            "replacement_rule": self.replacement_rule,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleDefinition":
        """Create from dictionary."""
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            category=RuleCategory(data["category"]),
            severity=Severity(data["severity"]),
            version=data["version"],
            description=data["description"],
            examples=data["examples"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            deprecated=data.get("deprecated", False),
            replacement_rule=data.get("replacement_rule"),
        )


class RuleRegistry:
    """
    Rule registry - Centralized rule management.

    Responsible for registration, lookup, and version management of all rules.
    """

    def __init__(self) -> None:
        self._rules: dict[str, RuleDefinition] = {}
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """Load default rules."""
        default_rules = [
            # Security rules (NFR-5 example)
            RuleDefinition(
                rule_id="SEC001",
                name="SQL Injection Vulnerability",
                category=RuleCategory.SECURITY,
                severity=Severity.HIGH,
                version="1.0.0",
                description="Security vulnerability where user input is directly inserted into SQL queries",
                examples=[
                    'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")',
                    "query = f\"DELETE FROM table WHERE col = '{value}'\"",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="SEC002",
                name="XSS Vulnerability",
                category=RuleCategory.SECURITY,
                severity=Severity.HIGH,
                version="1.0.0",
                description="Cross-Site Scripting vulnerability where user input is directly output to HTML",
                examples=[
                    'return f"<div>{user_input}</div>"',
                    "response.write(request.GET['param'])",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="PERF001",
                name="N+1 Query Pattern",
                category=RuleCategory.PERFORMANCE,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="Inefficient pattern where individual queries are executed within a loop",
                examples=[
                    "for user in users: user_data = db.query(UserData).filter(id=user.id).first()",
                    "for item in items: category = Category.query.get(item.category_id)",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="QUAL001",
                name="Missing Type Hints",
                category=RuleCategory.QUALITY,
                severity=Severity.LOW,
                version="1.0.0",
                description="Type hints are not provided for function parameters or return values",
                examples=[
                    "def process_data(data): return data * 2",
                    "def get_user(id): return User.query.get(id)",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="DOC001",
                name="Undocumented public API",
                category=RuleCategory.DOCUMENTATION,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="Documentation missing for public interface",
                examples=[
                    "def api_endpoint(request): pass  # undocumented",
                    "class PublicService: def method(self): pass  # undocumented",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
            RuleDefinition(
                rule_id="SCHEMA001",
                name="Missing Index",
                category=RuleCategory.SCHEMA,
                severity=Severity.MEDIUM,
                version="1.0.0",
                description="Index is missing on columns frequently used in WHERE clauses",
                examples=[
                    "WHERE user_id = ?  # user_id missing index",
                    "ORDER BY created_at DESC  # created_at missing index",
                ],
                created_at="2025-01-15",
                updated_at="2025-01-15",
            ),
        ]

        for rule in default_rules:
            self._rules[rule.rule_id] = rule

    def register_rule(self, rule: RuleDefinition) -> None:
        """Register a new rule."""
        if rule.rule_id in self._rules:
            raise ValueError(f"Rule {rule.rule_id} already exists")
        self._rules[rule.rule_id] = rule

    def get_rule(self, rule_id: str) -> RuleDefinition | None:
        """Get rule by ID."""
        return self._rules.get(rule_id)

    def get_rules_by_category(self, category: RuleCategory) -> list[RuleDefinition]:
        """Filter rules by category."""
        return [rule for rule in self._rules.values() if rule.category == category]

    def get_rules_by_severity(self, severity: Severity) -> list[RuleDefinition]:
        """Filter rules by severity."""
        return [rule for rule in self._rules.values() if rule.severity == severity]

    def validate_rule_id(self, rule_id: str) -> bool:
        """Validate if rule ID is valid."""
        return rule_id in self._rules

    def get_all_rules(self) -> dict[str, RuleDefinition]:
        """Get all rules."""
        return self._rules.copy()

    def export_rules(self) -> str:
        """Export rules to JSON format."""
        rules_dict = {rule_id: rule.to_dict() for rule_id, rule in self._rules.items()}
        return json.dumps(rules_dict, indent=2, ensure_ascii=False)

    def import_rules(self, json_str: str) -> None:
        """Import rules from JSON string."""
        rules_dict = json.loads(json_str)
        for rule_id, rule_data in rules_dict.items():
            rule = RuleDefinition.from_dict(rule_data)
            self._rules[rule_id] = rule

    def get_rule_version(self, rule_id: str) -> str | None:
        """Get current version of a rule."""
        rule = self.get_rule(rule_id)
        return rule.version if rule else None

    def is_rule_deprecated(self, rule_id: str) -> bool:
        """Check if rule is deprecated."""
        rule = self.get_rule(rule_id)
        return (
            rule.deprecated if rule else True
        )  # Consider non-existent rules as deprecated


# Global rule registry instance
rule_registry = RuleRegistry()


def get_rule_registry() -> RuleRegistry:
    """Get global rule registry instance."""
    return rule_registry


def validate_evidence_rule_compliance(rule_id: str) -> bool:
    """
    Validate if Evidence rule ID is valid (NFR-5 compliance).

    Args:
        rule_id: Rule ID to validate

    Returns:
        True if rule is valid, False otherwise
    """
    return rule_registry.validate_rule_id(rule_id)


def get_rule_info(rule_id: str) -> dict[str, Any] | None:
    """
    Get detailed information about a rule.

    Args:
        rule_id: Rule ID to look up

    Returns:
        Rule information dictionary or None if not found
    """
    rule = rule_registry.get_rule(rule_id)
    return rule.to_dict() if rule else None
