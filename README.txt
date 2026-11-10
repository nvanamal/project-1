# HgCdTe Nanoinformatics: Project 1

## Paper 1 – Machine-Learning Prediction of MWIR Absorbance

This repository contains the analysis code supporting **Machine-Learning Prediction of MWIR Absorbance in Plasmonic HgCdTe Photodetectors**.

- **Status:** Submitted to *Infrared Physics & Technology*
- **Manuscript No.:** `INFPHY-D-26-02527`

## Purpose

This study investigates how well machine-learning models predict mid-wave infrared (MWIR) absorptance in plasmonic HgCdTe photodetector configurations.

The analysis includes:

- Repeated random-row validation
- Complete-wavelength holdout validation
- Leave-one-design-out validation
- Feature-set ablation
- Grouped permutation importance
- Learning-curve analysis
- Reproducibility metadata

## Main Analysis Code

[`project1_paper1_hgcdte_full_analysis.py`](./project1_paper1_hgcdte_full_analysis.py)

## Data and Confidentiality Notice

This public repository contains non-confidential analysis code. The non-public optical-property inputs used in the electromagnetic simulations, including wavelength-dependent refractive-index (`n`) and extinction-coefficient (`k`) data, are excluded and are unavailable for redistribution by the author. The source COMSOL model and non-public material-property files are also excluded.

## Research Program and Related Studies

Paper 1 covers forward prediction. The broader HgCdTe nanoinformatics research program extends this work to inverse design, explainable AI, generative design, and multiple material systems.

### Paper 2 – Verification-First Inverse Design

**Verification-First Machine-Learning-Assisted Inverse Design of Plasmonic Nanostructures for HgCdTe Infrared Photodetectors**

- **Status:** Submitted to *Infrared Physics & Technology*
- **Manuscript No.:** `INFPHY-D-26-02529`

### Paper 3 – Explainable AI for HgCdTe Design

- **Research direction:** Use interpretable machine learning to identify design rules and evaluate them through physics-based verification.
- **Status:** Work in progress

### Paper 4 – Generative Nanoinformatics

- **Research direction:** Use generative methods to propose new HgCdTe nanostructures and verify candidate designs.
- **Status:** Work in progress

### Paper 5 – Multi-Material Nanoinformatics Platform

- **Research direction:** Extend the nanoinformatics workflow to multiple material systems.
- **Status:** Work in progress

## Repository Scope

This repository contains analysis code for **Paper 1 only**. Code, datasets, results, and supplementary materials for Papers 2–5 are outside its scope.



details of PROJECT 1, PAPER 1 — HgCdTe MACHINE-LEARNING ANALYSIS

File
----
project1_paper1_hgcdte_full_analysis.py

Purpose
-------
This single-file pipeline evaluates machine-learning prediction of MWIR
absorbance in plasmonic HgCdTe photodetector configurations. It includes:

- repeated random-row validation;
- complete-wavelength holdouts;
- leave-one-design-out validation;
- feature-set ablation;
- grouped permutation importance; and
- CSV results, figures, and run metadata.

Requirements
------------
Python 3.10 or newer is recommended.

Install the required packages:

    py -3.11 -m venv .venv
    .venv\Scripts\activate
    pip install numpy pandas scikit-learn matplotlib

Quick synthetic smoke test
--------------------------

    python project1_paper1_hgcdte_full_analysis.py --demo --quick

The demo dataset is synthetic and must not be reported as a research result.

Real research data
------------------
Place hgcdte_absorbance.csv in the same folder, or provide its full path:

    python project1_paper1_hgcdte_full_analysis.py --data hgcdte_absorbance.csv

Expected CSV columns
--------------------
geometry
primary_dimension
wavelength
primary_dimension_type
secondary_dimension
secondary_dimension_type
absorbance

Absorbance must be stored as a fraction between 0 and 1.

Outputs
-------
By default, results are written to project1_outputs/:

- all_repeated_results.csv
- model_comparison.csv
- lodo_results.csv
- fixed_tests_and_ablation.csv
- grouped_permutation_importance.csv
- run_metadata.json
- figures/model_comparison.png
- figures/lodo_mae.png
- figures/grouped_importance.png

Data note
---------
The original 280-row COMSOL table is not included in this package. Confirm
ownership, licensing, and publication permissions before placing research data
in a public GitHub repository.

Scientific caution
------------------
Random-row results alone do not establish generalization to unseen device
designs. Complete-wavelength and leave-one-design-out evaluations should be
reported alongside conventional random splits.

Author
------
Nagendrababu Vanamala, Ph.D.
