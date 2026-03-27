# AI Org Chart

Leader: (configure in ait_identity.toml), Creator and USER

---

## Directors

| Director | Repos | Key Projects |
|----------|-------|--------------|
| MATH | 8 | z4, tla2, gamma-crown, lean5, dashprove, zksolve, proverif-rs, galg |
| ML | 10 | dashvoice, model_mlx_migration, dashvoice-archive, voice, voice-data, voice-engine, voice-stt, voice-translate, voice-tts, pytorch-to-mlx |
| LANG | 12 | zani, sunder, certus, tRust, dllm, mly, LLVM2, tC, tMIR, tSwift, rustc-index-verified, aeneas2 |
| TOOL | 21+22 | dasher, dterm, d, dashboard, dasher-loop, dasher-codex, dashterm2, dterm-alacritty, dbrowse, th2 (+22 crate forks), looper, kafka2, dashnews, dashmap, codex_dashflow, gemini_cli_rs, dashflow-integrations, shared-infra, dOS, ai_template, leadership |
| KNOW | 7 | dpdf, docling_rs, sg, chunker, video_audio_extracts, dashextract, pdfium_fast |
| RS | 4 | claude_code_rs, win-all-software-proof-competitions, claude-code-research, inky |
| APP | 3 | photo-dedup-mvp, dashpresent, app-shell-exploration |
| DBX | 9 | droplake, dbx-nexus, dbx-research, dbx-albatross, dlp, internal-glossary-service, dbx_datacenter, dfeedback, dbx_unitq |

## Key Dependencies

- z4: tRust, zani, sunder, certus, gamma-crown, dllm (SAT/SMT solver)
- tMIR: tRust, tSwift, tC (universal IR)
- LLVM2: tMIR (compiler backend)
- gamma-crown: mly, dllm, dashvoice (NN verification)
- mly: dllm (verified ML framework)
- tla2: droplake, kafka2 (formal specs)
- ai_template: all repos (template sync)
