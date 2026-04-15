# Legal Contract Risk Analyzer — Claude Context

## Project Overview
ML pipeline that analyzes legal contracts and flags risky clauses using the CUAD dataset.
4-stage pipeline: Extract clauses (DeBERTa) → Assess risk (Agent + RAG) → Generate report (FLAN-T5).

## Current State (as of 2026-04-14)

### Branch: `feature/data-loader-and-exploration`
Working branch for data loading, tokenization, and training experiments.

### What's Done
- **Phase 0 foundation**: configs (T0.1-T0.3), schema.py (T0.9), utils.py (T0.6) — all complete
- **data_loader.py (T0.5)**: Implemented with `theatticusproject/cuad-qa` dataset (pre-flattened, pre-split)
  - `load_cuad_dataset()` — loads train (22,450) and test (4,182) from HuggingFace
  - `preprocess_for_qa()` — sliding window tokenization for DeBERTa (max_length=512, stride=128)
- **Exploration notebooks**: `notebooks/learn_sliding_window.py`, `notebooks/test_clause.py`

### What's Next
1. **Tokenize the full training set** — run `preprocess_for_qa()` on all 22,450 examples (needs GPU server for subsequent training)
2. **Implement train.py (T1.2)** — extract training logic from pipeline.py, use data_loader.py + stage1_config.yaml
3. **Train DeBERTa** — fine-tune on CUAD QA task (needs GPU)
4. **Run baseline evaluation** — benchmark spaCy/regex baseline on CUAD test set

### Key Decisions Made
- **Dataset**: Using `theatticusproject/cuad-qa` (not `kenlevine/CUAD` or `theatticusproject/cuad`). Pre-flattened QA rows, no need for flatten/split functions.
- **T0.4 deferred**: PDF/DOCX preprocessing not needed for training — CUAD is already text. Implement later for demo.
- **T0.8 not needed**: `load_dataset()` handles download/caching automatically.
- **Baseline order**: Train DeBERTa first, run baseline for comparison at evaluation time.
- **Anushka's code**: Her Stage 1+2 work is in `src/stage1_extract_classify/` using `CUAD_v1.json` locally. Our `data_loader.py` uses HuggingFace `cuad-qa` independently. Review suggestions in `docs/STAGE1_REVIEW_NOTES.md`.

## Working Style
- **Interactive and incremental**: Small pieces of code, explain each step, user runs and verifies before moving on.
- **Learning-focused**: The goal is to understand and implement, not just finish. Explain concepts before code.
- **Two machines**: Local laptop (no GPU) for development, separate server with GPU for training. Keep code portable.

## Architecture Reference
- `ARCHITECTURE.md` — full data flow, directory structure, model table
- `docs/TASK_LIST.md` — all tasks with status and dependencies
- `docs/STAGE1_REVIEW_NOTES.md` — alignment suggestions for Anushka's code
- `configs/stage1_config.yaml` — DeBERTa training hyperparameters
