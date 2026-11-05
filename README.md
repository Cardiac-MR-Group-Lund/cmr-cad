# Machine Learning Reveals Prognostic Value of Medical History in Suspected Coronary Artery Disease

This repository contains the exact analysis scripts used in the study
*Machine Learning Reveals Prognostic Value of Medical History in Suspected
Coronary Artery Disease*. It is published to support peer-review inspection,
methodological transparency, and algorithmic reproducibility.

The repository documents cohort selection, feature construction, validation,
model fitting, evaluation, and sensitivity analyses. It does not contain
patient-level data or results.

## Data Availability

Individual patient data and Swedish national registry records cannot be shared
publicly because of patient integrity, ethical regulations, and legal
restrictions. The `data/` and `results/` directories are intentionally
excluded from version control. The full pipeline requires approved access to
the source data in a controlled environment.

The committed files in `config/` define frozen feature and
cross-validation specifications used by the study. They contain
methodological and aggregate metadata only.

## Environment Setup

Use Python 3.12 and install the dependencies listed in
[`requirements.txt`](requirements.txt). `scikit-survival` may require
platform-specific compiled dependencies; consult its installation
documentation if pip cannot provide a compatible wheel.

### Linux or macOS (bash)

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Analysis Workflow

Run scripts from the repository root in the following numbered order as
required by the study analysis plan:

1. `01_audit_patient_v15.py` audits the private source extract using
   aggregate outputs only.
2. `01b_audit_ischemia_and_endpoints.py` and
   `01c_validate_hard_mace.py` verify ischemia and endpoint definitions.
3. `02_build_analysis_dataset.py` performs cohort selection and constructs
   the analysis dataset.
4. `03_freeze_registry_vocabulary.py` constructs the registry feature matrix
   specification, and `04_freeze_outer_cv_folds.py` fixes the outer
   cross-validation assignments.
5. `05_generate_table1.py` and `06_generate_kaplan_meier.py` produce
   descriptive cohort summaries and survival analyses.
6. `07_run_binary_rf_nested_cv.py` performs nested cross-validation and
   hyperparameter tuning for the binary Random Forest models.
7. `08_run_binary_endpoint_sensitivity.py` through
   `14_run_hard_mace_rsf_fixed.py` evaluate binary endpoint and survival-model
   sensitivity analyses, including Ridge Cox and Random Survival Forest
   models.
8. `15_generate_binary_calibration.py` through
   `26_run_mice_sensitivity_analysis.py` generate calibration analyses,
   diagnostic checks, comparison statistics, manuscript figures and tables,
   and multiple-imputation sensitivity analyses.

Each script accepts command-line paths for its private inputs and outputs.
Do not copy protected registry data, patient-level intermediate files, or
generated results into this repository.

## Citation

Permlid M, Karlsson M, Akil Engblom S, Ostenfeld E, Kjellström B, Kellman P,
Arheden H, Kraen M, Engblom H. Machine Learning Reveals Prognostic Value of
Medical History in Suspected Coronary Artery Disease. *JACC Adv.*
2026;:103443. DOI: [10.1016/j.jacadv.2026.103443](https://doi.org/10.1016/j.jacadv.2026.103443)

## License

This project is released under the [MIT License](LICENSE).
