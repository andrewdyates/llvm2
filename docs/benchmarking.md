# Benchmarking Guide

How to set up and run reproducible benchmark evaluations.

## Quick Start

```bash
# Copy templates to your project
cp -r benchmarks/templates/* benchmarks/

# Run evaluation
./benchmarks/run_eval.sh --suite default --run-id myrun
```

## Directory Structure

```
benchmarks/
├── templates/           # Copy these to start
│   ├── Dockerfile      # Container definition
│   ├── docker-compose.yaml
│   └── run_eval.sh     # Entry point
├── data/               # Input data (read-only)
├── logs/               # Execution logs
│   ├── build_images/   # Docker build output
│   └── run_evaluation/ # Per-run logs
└── results/            # Output results
    └── <run_id>/       # Results per run
        ├── metadata.json
        └── results.json
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_ID` | timestamp | Unique run identifier |
| `TIMEOUT` | 300 | Per-test timeout (seconds) |
| `MAX_WORKERS` | 4 | Parallel worker count |
| `SUITE` | default | Benchmark suite name |
| `LOGS_DIR` | benchmarks/logs | Log output directory |
| `RESULTS_DIR` | benchmarks/results | Results output directory |

### Command Line

```bash
./benchmarks/run_eval.sh --run-id myrun --timeout 60 --workers 8 --suite full
./benchmarks/run_eval.sh --list-suites
./benchmarks/run_eval.sh --help
```

## Docker Evaluation

For reproducible containerized runs:

```bash
# Build and run
cd benchmarks
docker compose up --build

# With custom settings
RUN_ID=experiment1 TIMEOUT=120 docker compose up --build

# Build only (for CI caching)
docker compose build
```

### Resource Requirements

Default limits (adjust in docker-compose.yaml):
- CPU: 8 cores
- Memory: 16GB
- Storage: varies by benchmark suite

## Output Format

### metadata.json

```json
{
    "benchmark_name": "project-bench",
    "run_id": "20260125-143022",
    "suite": "default",
    "timeout": 300,
    "max_workers": 4,
    "started_at": "2026-01-25T14:30:22Z",
    "completed_at": "2026-01-25T14:35:47Z",
    "commit": "abc123",
    "branch": "main",
    "run_context": {
        "machine_id": "runner-01",
        "cpu_model": "Intel(R) Xeon(R) ...",
        "cpu_arch": "x86_64",
        "os": "Linux",
        "os_version": "5.15.0-84",
        "container_image": "ghcr.io/org/bench:2026-01-29",
        "tool_version": "bench-runner 1.2.3"
    }
}
```

Optional `run_context` captures machine and environment details for comparing results across hardware or execution environments. (Source: Andrew Yates, Benchmark Run Context Metadata, designs/2026-01-29-benchmark-run-context-metadata.md:10-49)

Recommended fields include `machine_id`, `cpu_model`, `cpu_arch`, `os`, `os_version`, `container_image`, and `tool_version`. (Source: Andrew Yates, Benchmark Run Context Metadata, designs/2026-01-29-benchmark-run-context-metadata.md:35-50)

Populate `run_context` by extending your runner or `run_eval.sh`, and keep it optional so existing consumers can ignore it. (Source: Andrew Yates, Benchmark Run Context Metadata, designs/2026-01-29-benchmark-run-context-metadata.md:52-59)

### results.json

```json
{
    "suite": "default",
    "passed": 42,
    "failed": 3,
    "skipped": 5,
    "total": 50,
    "duration_seconds": 325,
    "tests": [
        {"name": "test_example", "status": "passed", "duration": 1.2},
        {"name": "test_slow", "status": "failed", "duration": 30.0, "error": "timeout"}
    ]
}
```

## Customization

1. **Edit `run_eval.sh`**: Implement `run_evaluation()` function
2. **Define suites**: Update `list_suites()` and suite handling
3. **Update Dockerfile**: Add project-specific dependencies
4. **Adjust resources**: Modify docker-compose.yaml limits

## Best Practices

1. **Pin versions**: Use specific image tags in Dockerfile
2. **Unique run IDs**: Include timestamp or commit hash
3. **Store logs**: Keep build and run logs for debugging
4. **Reproducibility**: Document all environment variables used
5. **Resource limits**: Set memory/CPU limits to prevent runaway tests

## Integration with CI

```bash
# CI pipeline example
export RUN_ID="ci-${CI_COMMIT_SHA:0:8}"
./benchmarks/run_eval.sh --suite quick --timeout 60

# Upload results
gh release upload "v${VERSION}" "benchmarks/results/${RUN_ID}/results.json"
```

## References

- SWE-bench evaluation: https://github.com/princeton-nlp/SWE-bench
- Docker best practices: https://docs.docker.com/develop/develop-images/dockerfile_best-practices/
