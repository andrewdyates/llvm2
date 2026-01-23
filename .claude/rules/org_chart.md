# Org Chart

> Minimal routing reference. Full details: `gh repo list ayates_dbx --no-archived`

## Directors

| Director | Mission | Key Repos | Message When |
|----------|---------|-----------|--------------|
| **MATH** | Proofs, verification | z4, tla2, gamma-crown | Need SMT, proofs, NN verification |
| **ML** | Machine learning, models, voice, | model_mlx_migration, voice | Need STT/TTS/LLM |
| **LANG** | Compilers, transpilation | tRust, tSwift | Need verified codegen |
| **TOOL** | Workflows, terminals | dashflow, dterm | Need orchestration, dev tools |
| **KNOW** | Documents, extraction | docling_rs | Need PDF/doc parsing |
| **RS** | Research, ports | inky, langchain_rs | Need reference implementations |
| **APP** | Products | dashpresent | Have features ready for users |

## Dependencies

```
z4 (5)         ← tSwift, tRust, kani_fast, tla2, gamma-crown
tRust (3)      ← tSwift, tla2, mly
gamma-crown(3) ← voice, mly, tC
dashflow (~5)  ← dasher, inky, dterm
```
