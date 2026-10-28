# Multi-Agent Admission Analysis System

Thesis research code for an LLM-based system that extracts structured transcript data, loads admission requirements, and evaluates applicant–program compatibility.

## Project Structure

- `agent1.py`: Extracts transcript information from PDFs and images.
- `agent2.py`: Loads program requirements from CSV files, with optional DAAD web scraping.
- `agent3.py`: Combines transcript data and admission requirements to generate a compatibility assessment.
- `experiments/`: Evaluation methods, model comparisons, ablation studies, and supporting analysis.
- `examples/`: Individual component examples and an integrated workflow.
- `tests/`: Original project check scripts; some require network access, API credentials, or private data.

## Installation and API Credentials

Python 3.10 or later is recommended. Install dependencies in a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
export GEMINI_API_KEY="$GOOGLE_API_KEY"
export OPENAI_API_KEY="your-openai-api-key"
```

`.env.example` is a configuration reference only; the code does not automatically load `.env` files. PDF-to-image conversion also requires Poppler. Web scraping requires Chrome and a compatible browser driver.

## Data and Reproducibility

This prepared copy excludes real transcripts, identity mappings, admission records, individual experiment results, scraped datasets, and thesis documents. The original experiments depend on these local inputs, so this directory alone cannot reproduce the thesis results. Some scripts retain historical data paths; supply data you are authorized to process and adjust the paths before running them.

Example API usage with your own input:

```python
import os
from agent1 import TranscriptExtractor

extractor = TranscriptExtractor(os.environ["GOOGLE_API_KEY"])
result = extractor.extract_transcript("/path/to/authorized-transcript.pdf")
```

This operation sends transcript content to the model service. Confirm authorization to process and transmit the data before running it. Model names follow the historical experiment configuration; check their availability before use. Outputs are intended for research and do not constitute official admission decisions. The credit conversion factor in the code is an experimental assumption.

## Publication Status and Licensing

This is a prepared copy for review and has not been published as part of this cleanup. Code ownership, university or collaborator publication requirements, and an appropriate open-source license still need to be confirmed by the author. No open-source license has been added on the author's behalf. Third-party dependencies remain subject to their respective licenses; their source code is not bundled here.

Real data and thesis materials are excluded. Do not add personal transcripts, credentials, data without redistribution permission, or thesis attachments to a public repository. The review records are stored separately in `整理审查报告.md` and `文件处置清单.csv`.

## Running Examples and Experiments

Run module commands from the project root so that imports and relative data paths resolve consistently:

```sh
python -m examples.example_integrated_agents
python -m experiments.batch_model_comparison --help
python -m experiments.compare_system_vs_baseline --help
```

Examples and experiments require your own authorized data and environment variables; they are not demonstrations that run without input data. The original data shortcuts have been removed. Default paths such as `Student_Transcripts/`, `acceptance.json`, and `university_programs.csv` must be configured locally. Do not commit real personal data.

Run check scripts from the project root using `python -m tests.<module_name>`. These are the original manual check scripts, not a verified automated test suite.

The cleanup removed legacy experiment entry points, the old GPT-4o baseline, and one-off result repair and rerun tools. API integration remains available, with credentials supplied through environment variables. No real API keys are included.
