# Changelog

All notable changes to ai_template will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
after reaching V1.

## [Unreleased]

### Added
- V1 readiness check in `vision_health.py --v1-check` (#2115)
- SWE-bench evaluation runner in `evals/templates/swe_bench.py` (#2128)
- Multi-day trend analysis in pulse metrics (#2136)

### Changed
- Refactored pulse.py into package structure `ai_template_scripts/pulse/` (#404)
- Refactored checkpoint.py into `looper/checkpoint/` package (1,046 → 4 modules) (#2079)

### Deprecated
- None

### Removed
- `pulse_monolith.py` - replaced by pulse package

### Fixed
- Looper sync docs: added legacy key documentation and precedence rules (#2212)
- AI_CODER env var and tool version documentation (stale #2213 - implemented in 519f3a83)
- PreToolUse security hooks confirmed deployed (stale #1847 - check_security.py active)
- git wrapper .pid_worker issue (closed #2231 as duplicate of #2218)
- validate_numeric_claim missing tests and tolerance validation (#2149)
- SWE-bench: FAIL_TO_PASS/PASS_TO_PASS resolution tracking (#2156)
- python3 wrapper to block inline Python for issue filtering (#2159)
- SWE-bench test coverage - comprehensive tests added (#2150)
- SWE-bench multi-strategy patch application (#2157)
- SWE-bench agent prompt for patch output (#2148)
- IndexError in issue_manager_audit.py closing hash parsing (#2165)
- SWE-bench test timeout split causing false failures (#2166)
- evals/ not importable as Python package - missing __init__.py (#2160)
- check_deps.py hardcoded ~/ai_template paths (#2162)
- load_dataset mock target for tests when datasets unavailable (#2163)

### Security
- None

## [0.1.0] - 2026-02-03

Initial CHANGELOG creation for V1 deprecation policy tracking.

### Added
- Template infrastructure for AI-driven development
- Looper automation system for autonomous AI sessions
- Sync system for propagating template changes to downstream repos
- Pulse health metrics collection
- GitHub API wrapper with rate limiting
- Vision health check for success criteria tracking

### Notes
- ai_template is currently at **USABLE** readiness level
- V1 readiness requires: API stability, deprecation policy, test coverage, documentation accuracy
- See `designs/2026-02-03-v1-success-criteria.md` for V1 criteria

[Unreleased]: https://github.com/ayates_dbx/ai_template/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ayates_dbx/ai_template/releases/tag/v0.1.0
