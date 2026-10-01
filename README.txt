PROJECT 1, PAPER 1 — HgCdTe MACHINE-LEARNING ANALYSIS

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
