# Modular Bayesian transmission analysis for the 2026 DRC Bundibugyo virus disease outbreak

This repository contains the analysis code, harmonised aggregate data, model outputs, and figure-generation scripts for a modular Bayesian analysis of the 2026 Bundibugyo virus disease outbreak in the Democratic Republic of the Congo.

The workflow links:

1. Event-time reconstruction of publicly reported aggregate case and death series
2. Module 1: renewal-informed state-space inference for time-varying Rt
3. Module 2: posterior-constrained SEIHFR reconstruction and pathway decomposition
4. Forward 90-day response scenarios and delayed-response analysis
5. Global sensitivity and robustness analyses

No individual-level or personally identifiable data are included.

## Repository structure

```text
config/             Model and analysis configuration
data/               Harmonised aggregate source data and source registry
figures/            Final main and supplementary figures
results/            Final model outputs and posterior summaries
scripts/            Utility scripts for supplementary figure generation
source_evidence/    Archived public source evidence and source manifest
src/                Analysis modules
00_run_full_project.py
run_publication_robustness.py
validate_project.py
```

## Requirements

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

## Reproducing the main analysis

The default configuration is set to publication mode. To rerun the full workflow:

```bash
python 00_run_full_project.py
```

## Robustness audit

Three random-seed robustness runs can be reproduced with:

```bash
python run_publication_robustness.py
```

Outputs are written to `results/publication_seed_robustness.csv` and `results/publication_seed_robustness_summary.json`.

## Validation

To check that expected outputs are present and internally consistent:

```bash
python validate_project.py
```

## Data sources

The analysis uses publicly available aggregate outbreak reports from the DRC Ministry of Health/INSP, WHO, WHO Regional Office for Africa, Africa CDC, and ECDC. The file `data/source_registry.csv` records the source hierarchy and URLs. Archived public source captures used during verification are provided in `source_evidence/`.

Derived fields such as case shares, crude case-fatality ratios, reconstructed event-time incidence, and bed-occupancy percentages were calculated from aggregate source data.
