# AVerImaTeC Shared Task - Project Context

## Project Overview

This is the baseline implementation for the **AVerImaTeC shared task** - a competition/research dataset for **Automatic Verification of Image-Text Claims with Evidence from the Web**. The shared task is part of FEVER9 workshop co-located with EACL2026.

**Paper**: [AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims](https://arxiv.org/pdf/2505.17978)
**Leaderboard**: https://huggingface.co/spaces/FEVER-IT/AVerImaTeC
**Dataset**: https://huggingface.co/datasets/Rui4416/AVerImaTeC

### Core Task
Given an image-text claim, the system must:
1. Generate relevant questions to verify the claim
2. Retrieve evidence from a knowledge store (or web search)
3. Generate answers with supporting evidence (text + images)
4. Classify the claim verdict (Supported/Refuted/Not Enough Evidence)
5. Generate justification for the verdict

## Project Structure

```
/mnt/personal/ullriher/aic_averimatec/
├── src/                          # Main source code
│   ├── pipeline.py              # Main pipeline orchestration
│   ├── mm_checker.py            # Multi-modal fact checker (core logic)
│   ├── evidence_generation.py  # Evidence generation components
│   ├── retrieval.py            # Evidence retrieval
│   ├── classification.py       # Verdict classification
│   ├── averitec.py            # Data structures and utilities
│   ├── config.py              # Command-line argument parser
│   ├── labels.py              # Label definitions
│   └── dynamic_mm_fc/         # Dynamic multi-modal fact checking modules
│       ├── qg_model.py        # Question generation model
│       ├── qa_model.py        # Question answering model
│       ├── verifier.py        # Answer verification
│       ├── planner.py         # Question planning
│       ├── justification_gen.py  # Justification generation
│       ├── summarizer.py      # Evidence summarization
│       ├── tools.py           # Helper tools
│       ├── utils.py           # Utility functions
│       ├── templates/         # Prompt templates
│       ├── conv_utils/        # Conversion utilities (QA to evidence)
│       └── web_related/       # Web search utilities
├── templates/                 # Evaluation prompt templates
│   ├── evid_evaluation_joint.py    # Evidence evaluation (joint)
│   ├── evid_evaluation_text.txt    # Evidence eval prompt
│   ├── justi_evaluation_text.txt   # Justification eval prompt
│   ├── ques_evaluation_text.txt    # Question eval prompt
│   └── qa_to_evid_demos.txt       # QA to evidence demos
├── prepare_submission/        # Submission format conversion & evaluation
│   ├── eval_offline.py       # Offline evaluation script
│   ├── ref_eval.py          # Reference evaluation
│   ├── utils.py             # Evaluation utilities
│   └── ipython/             # Jupyter notebooks for result conversion
├── notebooks/                # Experimentation notebooks
├── script/                   # Shell scripts for training/evaluation
│   ├── baseline_script.sh   # Main baseline execution script
│   ├── eval_offline.sh      # Evaluation script
│   └── _slurm_job*.sbatch  # SLURM job scripts for HPC
├── private_info/            # API keys (GITIGNORED - DO NOT COMMIT)
│   └── API_keys.py         # GEMINI_API_KEY, GOOGLE_API_KEY, etc.
├── logs/                    # Training/execution logs
└── essential_requirement.txt  # Python dependencies
```

## Key Architecture Components

### 1. Pipeline (`src/pipeline.py`)
The main execution flow:
```
Datapoint → Retrieval → Evidence Generation → Classification → PipelineResult
```
- **Input**: `Datapoint` (claim + images)
- **Output**: `PipelineResult` with questions, evidence, verdict, justification

### 2. MM_Checker (`src/mm_checker.py`)
Multi-modal fact checker that implements the core verification logic:
- Question generation (dynamic, parallel, or hybrid strategies)
- Question answering using retrieved evidence
- Answer verification
- Justification generation
- Supports multiple MLLMs: Gemini 2.0, Qwen, Gemma, LLaVA

### 3. Evidence Generation (`src/evidence_generation.py`)
- Generates `Evidence` objects containing:
  - Question + Answer
  - Evidence text
  - Source URL
  - Scraped text
  - Supporting images
- Converts QA pairs to submission format

### 4. Question Generation Strategies
Three modes available (configured in `config.py`):
- **Dynamic QG** (default): Sequential question generation based on previous answers
- **Parallel QG** (`--PARA_QG True`): Generate all questions upfront
- **Hybrid QG** (`--HYBRID_QG True`): Combination of both approaches

## Configuration & Arguments

Key arguments in `src/config.py`:
- `--ROOT_PATH`: Project root directory (default: `/mnt/personal/ullriher/aic_averimatec`)
- `--TEST_MODE`: "val" or "test"
- `--LLM_NAME`: LLM for text-only tasks (options: "gemini-2.0-flash-001", "qwen", "gemma")
- `--MLLM_NAME`: Multi-modal LLM (options: "gemini-2.0-flash-001", "qwen", "gemma", "llava")
- `--MAX_QA_ITER`: Maximum Q&A iterations (default: 5)
- `--NUM_GEN_QUES`: Number of questions to generate (default: 8)
- `--MAX_NUM_IMAGES`: Max images per claim (default: 3)
- `--QG_ICL`: Use few-shot in-context learning for QG (default: False)
- `--TOOL_ICL`: Use ICL for tool usage (default: True)
- `--DEBUG`: Debug mode (test on few claims only)
- `--DATA_STORE`: Use provided knowledge store vs. web search
- `--DATASTORE_PATH`: Path to knowledge store

## Data & Knowledge Store

### Dataset Location
- **Images**: `data/data_clean/images/` (from images.zip)
- **JSON splits**: `data/data_clean/split_data/` (train.json, val.json, test.json)

### Knowledge Store
Pre-computed evidence to avoid expensive web searches:
- Download from: https://drive.google.com/drive/folders/1vjy7mjA4NTuLQfPh5-NZFpaxn8_H9rUs
- Set `--DATASTORE_PATH` to the download location

## Running the Baseline

### Main Execution Script
See `script/baseline_script.sh`:
```bash
python -m src.pipeline \
  --ROOT_PATH /path/to/project \
  --DATASTORE_PATH /path/to/knowledge_store \
  --LLM_NAME gemini-2.0-flash-001 \
  --MLLM_NAME gemini-2.0-flash-001 \
  --DEBUG True  # Remove for full run
```

### Evaluation
See `script/eval_offline.sh`:
```bash
python prepare_submission/eval_offline.py \
  --root_dir /path/to/project \
  --pred_file_path /path/to/predictions.json
```

Evaluation metrics:
- Question score (relevance to claim)
- Evidence score (faithfulness to sources)
- Conditional verdict accuracy
- Conditional justification score

## Submission Format

Convert baseline outputs to submission format using:
- `prepare_submission/ipython/Result_Convert.ipynb`

Required fields:
- `id`: claim ID
- `claim`: claim text
- `questions`: list of generated questions
- `evidence`: list of evidence objects (text, URL, images)
- `verdict`: "Supported" / "Refuted" / "Not Enough Evidence"
- `justification`: text justification for the verdict

## Technologies & Dependencies

### Core Libraries
- PyTorch 2.4.0+cu121
- Transformers 4.50.2 (Hugging Face)
- Accelerate 0.33.0
- Python 3.9.17

### NLP & ML
- NLTK 3.8.1
- NumPy 1.24.0
- SciPy (for softmax)
- rank-bm25 (for BM25 retrieval)

### APIs
- Google Generative AI (Gemini)
- Google Search API
- OpenAI (for embeddings/fallback)

### Other
- inflect (for text processing)
- dirtyjson (for robust JSON parsing)

## Important Notes

### Security
- **NEVER commit `private_info/API_keys.py`** - it's now in .gitignore
- API keys required: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`

### Notebooks
- Jupyter notebooks in `notebooks/` are for experimentation
- **Always clear outputs before committing** to keep file sizes small
- Main notebooks:
  - `download_data.ipynb` - Data preparation
  - `pipeline_subquery.ipynb` - Pipeline testing
  - `reverse_image_search.ipynb` - Image search experiments

### HPC Execution
- SLURM scripts available in `script/` for running on HPC clusters
- Different variants for GPU types (H200, DGX-10) and CPU-only jobs

### Coding Conventions
- Dataclasses used for structured data (`@dataclass` decorator)
- Type hints extensively used (`typing` module)
- Modular design with clear separation of concerns
- JSON for data serialization, pickle for model caching

## Common Workflows

### 1. Running a Quick Test
```bash
python -m src.pipeline --DEBUG True --TEST_MODE val
```

### 2. Full Baseline Run
```bash
bash script/baseline_script.sh
```

### 3. Evaluate Results
```bash
bash script/eval_offline.sh
# Then use prepare_submission/ipython/Eval_Score_Compute.ipynb
```

### 4. Submit to Leaderboard
1. Run baseline to generate predictions
2. Convert using `Result_Convert.ipynb`
3. Upload to https://huggingface.co/spaces/FEVER-IT/AVerImaTeC

## Recent Changes

Based on git status, recent work focuses on:
- Evaluation pipeline refinements (`eval_offline.py`, `evid_evaluation_joint.py`)
- Evidence generation improvements (`evidence_generation.py`)
- Notebook experiments with pipeline and queries

## Tips for Claude

1. **When modifying evaluation code**: Check both `prepare_submission/eval_offline.py` and `templates/evid_evaluation_joint.py`
2. **When adding new MLLMs**: Update model initialization in `src/dynamic_mm_fc/` modules
3. **When debugging**: Use `--DEBUG True` flag to test on small subset
4. **When working with evidence**: The `Evidence` dataclass is central - modifications ripple through pipeline
5. **File sizes**: Be mindful of notebook outputs - clear them before committing
6. **Dependencies**: Virtual environment at `/mnt/personal/ullriher/venvs/aug25/`
