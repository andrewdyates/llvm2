# Success Metrics Methodology

Author: Andrew Yates <ayates@dropbox.com>

This document defines measurement methodology for each VISION.md success metric.
Each metric has: definition, measurement command, pass/fail threshold, evidence location, and current baseline.

**Phase-gating:** Metrics are tied to phases defined in VISION.md. Metrics marked with earlier phases are measurable now; others will fail until their phase is complete. This is expected.

**Evidence storage patterns:**
- Benchmark-style: `benchmarks/results/<run_id>/results.json`
- Eval-style: `evals/results/<eval_id>/<run_id>/{metadata,results}.json`
- Proof-style: `proofs/runs/<run_id>/manifest.json` + logs

---

## Functionality

### SM-FUNC-1

**[Describe primary functional capability]**

| Field | Value |
|-------|-------|
| Phase | N |
| Definition | [What this capability means in concrete terms] |
| Measurement | [Command or process to measure] |
| Threshold | [Pass/fail criteria] |
| Evidence | [Where results are stored] |
| Baseline | [Current state or N/A if not implemented] |

---

## Quality

### SM-QUAL-1

**[Describe quality attribute]**

| Field | Value |
|-------|-------|
| Phase | N |
| Definition | [What quality standard must be met] |
| Measurement | [Command or process to measure] |
| Threshold | [Pass/fail criteria with specific numbers] |
| Evidence | [Where results are stored] |
| Baseline | [Current state or N/A if not implemented] |

---

## Efficiency

### SM-EFF-1

**[Describe efficiency metric]**

| Field | Value |
|-------|-------|
| Phase | N |
| Definition | [What efficiency target must be met] |
| Measurement | [Command or process to measure] |
| Threshold | [Pass/fail criteria with specific numbers] |
| Evidence | [Where results are stored] |
| Baseline | [Current state or N/A if not implemented] |

**Verification command:**
```bash
# Example verification script
# command --with-flags
# Store results in benchmarks/results/$(date +%Y%m%d)/
```

---

## Correctness

### Evidence format (formal verification claims)

For any **formal verification** claim (Kani/Verus/TLA+/etc) used as evidence for SM-COR-*,
the evidence is considered acceptable only if it includes:

- **Commit**: git SHA of the code being verified
- **Tool versions**: verifier version (and language toolchain version if applicable)
- **Command**: the exact command line used (including harness/model name(s))
- **Result summary**: what passed/failed (counts, plus which harnesses/models)
- **Artifacts**: a stable-on-disk bundle containing raw logs (stdout/stderr) and a short machine-readable summary

Recommended artifact layout:
`proofs/runs/<run_id>/manifest.json` (fields above) plus `stdout.log` and `stderr.log`.

### SM-COR-1

**[Describe correctness property]**

| Field | Value |
|-------|-------|
| Phase | N |
| Definition | [What correctness property must hold] |
| Measurement | [Verification command] |
| Threshold | [All proofs pass; no failures] |
| Evidence | [Where proof artifacts are stored] |
| Baseline | [Current state or N/A if not implemented] |

---

## Summary Table

| ID | Metric | Phase | Measurable Now? |
|----|--------|-------|-----------------|
| SM-FUNC-1 | [Description] | N | No/Yes/Partial |
| SM-QUAL-1 | [Description] | N | No/Yes/Partial |
| SM-EFF-1 | [Description] | N | No/Yes/Partial |
| SM-COR-1 | [Description] | N | No/Yes/Partial |

---

## References

- `VISION.md` - Success Metrics section
- `docs/benchmarking.md` - Benchmark harness and result format (if applicable)
- `proofs/` - Formal verification artifacts (if applicable)
