# AGENTS.md - Project Knowledge (AVerImaTeC)

## Most Important Files (read first)
- `src/pipeline.py`: Main orchestration from datapoint -> retrieval -> evidence -> classification -> result.
- `src/mm_checker.py`: Core multi-modal fact checker; handles QG/QA/verification/justification and model routing.
- `src/evidence_generation.py`: Builds `Evidence` objects and converts QA pairs to submission-friendly evidence.
- `src/retrieval.py`: Evidence retrieval layer (datastore or web search).
- `src/classification.py`: Verdict prediction (Supported/Refuted/Not Enough Evidence).
- `src/averitec.py`: Core dataclasses and shared utilities for the pipeline.
- `src/config.py`: CLI args and defaults; defines QG modes and run-time settings.
- `src/labels.py`: Label definitions for verdicts and evaluation.
- `src/dynamic_mm_fc/`: Modular components for QG, QA, verification, planning, summarization, and prompt templates.

## Project Goal
Baseline system for the AVerImaTeC shared task (Automatic Verification of Image-Text Claims with Evidence from the Web). The pipeline generates questions, retrieves evidence, answers questions, verifies answers, classifies the claim, and outputs a justification.

## Tech Stack
- Python 3.9
- PyTorch + Transformers + Accelerate
- NLP tooling: NLTK, NumPy, SciPy, rank-bm25
- APIs: Google Generative AI (Gemini), Google Search, optional OpenAI embeddings

## File Map (quick navigation)
- `src/`
  - `pipeline.py` (orchestration)
  - `mm_checker.py` (core logic)
  - `evidence_generation.py`, `retrieval.py`, `classification.py`
  - `averitec.py` (dataclasses), `config.py` (CLI), `labels.py`
  - `dynamic_mm_fc/` (QG/QA/verification/summarization/planning, prompt templates, web tools)
- `templates/`: evaluation prompt templates for question/evidence/justification evaluation.
- `prepare_submission/`: offline evaluation and submission conversion utilities.
- `script/`: baseline and evaluation shell scripts (plus SLURM variants).
- `notebooks/`: experiments; clear outputs before committing.
- `private_info/`: API keys (gitignored, never commit).

## Data Locations
- Images: `data/data_clean/images/`
- Splits: `data/data_clean/split_data/` (train/val/test JSON)
- Optional datastore: external download; set via `--DATASTORE_PATH`.

## Common Commands
- Quick debug run:
  - `python -m src.pipeline --DEBUG True --TEST_MODE val`
- Baseline script:
  - `bash script/baseline_script.sh`
- Offline eval:
  - `bash script/eval_offline.sh`

## Configuration Notes
- QG modes in `src/config.py`:
  - Dynamic (default), Parallel (`--PARA_QG True`), Hybrid (`--HYBRID_QG True`)
- Model selection via `--LLM_NAME` and `--MLLM_NAME` (Gemini/Qwen/Gemma/LLaVA).

## Submission Output (key fields)
- `id`, `claim`, `questions`, `evidence` (text + URL + images), `verdict`, `justification`.

## Security & Hygiene
- Never commit `private_info/API_keys.py`.
- Clear notebook outputs before committing.

## Tips
- Evidence schema changes ripple through pipeline; check `evidence_generation.py` and `prepare_submission/` together.
- When editing evaluation logic, update both `prepare_submission/eval_offline.py` and `templates/evid_evaluation_joint.py`.
