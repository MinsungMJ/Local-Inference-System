"""LIS Inspect — post-execution TUI inspector for LIS runs.

Core surface: CLI argument parsing, JSON artifact load, stderr perf-stage and
perf-summary parse, merged InspectSession, Overview tab, Perf tab, Per-Token
tab, Artifact tab, Raw tab, Issues tab, and two-run compare mode.

JSON (`--report-json`) is canonical. stderr (`--stderr-log`) is supplementary.
"""

__version__ = "0.1.0"
