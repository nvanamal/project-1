#!/usr/bin/env python3
"""Single-file reproducibility pipeline for Project 1, Paper 1.

Expected real-data CSV columns:
    geometry, primary_dimension, wavelength, primary_dimension_type,
    secondary_dimension, secondary_dimension_type, absorbance

Examples:
    python project1_paper1_hgcdte_full_analysis.py --demo --quick
    python project1_paper1_hgcdte_full_analysis.py --data hgcdte_absorbance.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
CATEGORICAL = ["geometry", "primary_dimension_type", "secondary_dimension_type"]
NUMERIC = ["primary_dimension", "wavelength", "secondary_dimension"]
FULL = CATEGORICAL + NUMERIC
WAVELENGTH_ONLY = ["wavelength"]
GEOMETRY_ONLY = [
    "geometry",
    "primary_dimension",
    "primary_dimension_type",
    "secondary_dimension",
    "secondary_dimension_type",
]
REQUIRED = [
    "geometry",
    "primary_dimension",
    "wavelength",
    "primary_dimension_type",
    "secondary_dimension",
    "secondary_dimension_type",
    "absorbance",
]


def sha256(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_demo(path: Path) -> None:
    """Create a small synthetic dataset for a smoke test only."""
    rng = np.random.default_rng(7)
    waves = np.linspace(2.5, 5.0, 20)
    rows = []
    designs = [
        ("cube", 0.20, "width", np.nan, "none"),
        ("cube", 0.30, "width", np.nan, "none"),
        ("rectangle", 0.22, "width", 0.32, "length"),
        ("sphere", 0.20, "radius", np.nan, "none"),
        ("cylinder", 0.25, "radius", 0.20, "height"),
        ("cone", 0.25, "radius", 0.25, "height"),
        ("pyramid", 0.28, "width", 0.25, "height"),
        ("ring", 0.30, "outer_radius", 0.12, "inner_radius"),
        ("without ITO", np.nan, "none", np.nan, "none"),
    ]
    effects = dict(
        zip(
            [design[0] for design in designs],
            [0.03, 0.025, 0.015, 0.012, 0.008, -0.005, -0.01, -0.025, -0.06],
        )
    )

    for geometry, d1, t1, d2, t2 in designs:
        for wavelength in waves:
            spectral = (
                0.63
                + 0.08 * np.exp(-((wavelength - 4.05) / 0.55) ** 2)
                - 0.40 / (1 + np.exp(-(wavelength - 4.68) * 24))
            )
            absorbance = np.clip(
                spectral + effects[geometry] + rng.normal(0, 0.004), 0, 1
            )
            rows.append([geometry, d1, wavelength, t1, d2, t2, absorbance])

    pd.DataFrame(rows, columns=REQUIRED).to_csv(path, index=False)


def load_data(path: Path) -> pd.DataFrame:
    """Load, validate, normalize, and identify designs in the input table."""
    dataframe = pd.read_csv(path)
    aliases = {
        "Geometry": "geometry",
        "Wavelength": "wavelength",
        "Absorbance": "absorbance",
        "absorption": "absorbance",
    }
    dataframe = dataframe.rename(
        columns={key: value for key, value in aliases.items() if key in dataframe.columns}
    )

    missing = [column for column in REQUIRED if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    dataframe = dataframe[REQUIRED].copy()
    for column in NUMERIC + ["absorbance"]:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")

    if dataframe.wavelength.isna().any() or dataframe.absorbance.isna().any():
        raise ValueError("Wavelength/absorbance contains missing or nonnumeric values")
    if not dataframe.absorbance.between(0, 1).all():
        raise ValueError("Absorbance must be stored as a fraction in [0, 1]")
    if dataframe.duplicated().any():
        raise ValueError("Duplicate rows detected")

    def normalize(value: object) -> str:
        return "NA" if pd.isna(value) else str(value).strip().lower()

    design_columns = [
        "geometry",
        "primary_dimension",
        "primary_dimension_type",
        "secondary_dimension",
        "secondary_dimension_type",
    ]
    dataframe["design_id"] = dataframe.apply(
        lambda row: "|".join(normalize(row[column]) for column in design_columns), axis=1
    )
    return dataframe


def preprocessor(features: list[str], scale: bool = False) -> ColumnTransformer:
    """Build preprocessing for a selected feature set."""
    categorical = [column for column in CATEGORICAL if column in features]
    numeric = [column for column in NUMERIC if column in features]
    numeric_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=-1.0))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(numeric_steps), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="none"),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def models(features: list[str], seed: int, quick: bool = False) -> dict[str, Pipeline]:
    """Return all evaluated model pipelines."""
    forest_trees = 60 if quick else 500
    boosting_trees = 60 if quick else 300
    mlp_iterations = 250 if quick else 2500
    return {
        "Mean baseline": Pipeline(
            [("pre", preprocessor(features)), ("model", DummyRegressor(strategy="mean"))]
        ),
        "Ridge": Pipeline(
            [("pre", preprocessor(features, True)), ("model", Ridge(alpha=1.0))]
        ),
        "Random Forest": Pipeline(
            [
                ("pre", preprocessor(features)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=forest_trees,
                        max_features=0.8,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("pre", preprocessor(features)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=boosting_trees,
                        max_depth=2,
                        learning_rate=0.03,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "MLP": Pipeline(
            [
                ("pre", preprocessor(features, True)),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=0.001,
                        max_iter=mlp_iterations,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics."""
    return {
        "MAE": mean_absolute_error(y_true, prediction),
        "RMSE": mean_squared_error(y_true, prediction) ** 0.5,
        "R2": r2_score(y_true, prediction),
    }


def run(dataframe: pd.DataFrame, output: Path, quick: bool = False) -> None:
    """Run repeated validation, ablation, importance, and figure generation."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    rows = []
    random_repeats = 3 if quick else 25
    wavelength_repeats = 5 if quick else 50

    for repeat in range(random_repeats):
        train_index, test_index = train_test_split(
            np.arange(len(dataframe)), test_size=0.20, random_state=repeat
        )
        for name, model in models(FULL, 100 + repeat, quick).items():
            model.fit(dataframe.iloc[train_index][FULL], dataframe.iloc[train_index].absorbance)
            prediction = model.predict(dataframe.iloc[test_index][FULL])
            rows.append(
                {
                    "protocol": "random_rows",
                    "repeat": repeat,
                    "model": name,
                    **metrics(dataframe.iloc[test_index].absorbance, prediction),
                }
            )

    rng = np.random.default_rng(SEED)
    wavelengths = np.sort(dataframe.wavelength.unique())
    held_count = min(4, max(1, len(wavelengths) // 4))
    for repeat in range(wavelength_repeats):
        held = np.sort(rng.choice(wavelengths, held_count, replace=False))
        test_mask = dataframe.wavelength.isin(held)
        train_mask = ~test_mask
        for name, model in models(FULL, 200 + repeat, quick).items():
            model.fit(dataframe.loc[train_mask, FULL], dataframe.loc[train_mask, "absorbance"])
            prediction = model.predict(dataframe.loc[test_mask, FULL])
            rows.append(
                {
                    "protocol": "wavelength_holdout",
                    "repeat": repeat,
                    "held_out": ";".join(map(str, held)),
                    "model": name,
                    **metrics(dataframe.loc[test_mask, "absorbance"], prediction),
                }
            )

    lodo_rows = []
    for repeat, design in enumerate(sorted(dataframe.design_id.unique())):
        test_mask = dataframe.design_id == design
        train_mask = ~test_mask
        for name, model in models(FULL, 300 + repeat, quick).items():
            model.fit(dataframe.loc[train_mask, FULL], dataframe.loc[train_mask, "absorbance"])
            prediction = model.predict(dataframe.loc[test_mask, FULL])
            row = {
                "protocol": "LODO",
                "repeat": repeat,
                "withheld_design": design,
                "model": name,
                **metrics(dataframe.loc[test_mask, "absorbance"], prediction),
            }
            rows.append(row)
            lodo_rows.append(row)

    all_results = pd.DataFrame(rows)
    all_results.to_csv(output / "all_repeated_results.csv", index=False)
    summary = (
        all_results.groupby(["protocol", "model"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            R2_mean=("R2", "mean"),
            R2_median=("R2", "median"),
        )
        .reset_index()
    )
    summary.to_csv(output / "model_comparison.csv", index=False)
    pd.DataFrame(lodo_rows).to_csv(output / "lodo_results.csv", index=False)

    fixed_rows = []
    partitions = {
        "random": train_test_split(
            np.arange(len(dataframe)), test_size=0.2, random_state=SEED
        )
    }
    target_waves = np.array([2.6316, 3.5526, 4.0789, 4.2105])
    wavelength_mask = np.isclose(
        dataframe.wavelength.to_numpy()[:, None], target_waves[None, :], atol=0.03
    ).any(axis=1)
    if wavelength_mask.any() and (~wavelength_mask).any():
        partitions["fixed_wavelength"] = (
            np.where(~wavelength_mask)[0],
            np.where(wavelength_mask)[0],
        )

    excluded_category = dataframe.geometry.astype(str).str.lower().isin(
        ["ring", "cone", "pyramid"]
    )
    if excluded_category.any() and (~excluded_category).any():
        partitions["excluded_category"] = (
            np.where(~excluded_category)[0],
            np.where(excluded_category)[0],
        )

    feature_sets = [
        ("full", FULL),
        ("wavelength_only", WAVELENGTH_ONLY),
        ("geometry_dimensions_only", GEOMETRY_ONLY),
    ]
    for protocol, (train_index, test_index) in partitions.items():
        for feature_name, features in feature_sets:
            model = models(features, SEED, quick)["Random Forest"]
            model.fit(
                dataframe.iloc[train_index][features],
                dataframe.iloc[train_index].absorbance,
            )
            prediction = model.predict(dataframe.iloc[test_index][features])
            fixed_rows.append(
                {
                    "protocol": protocol,
                    "feature_set": feature_name,
                    **metrics(dataframe.iloc[test_index].absorbance, prediction),
                }
            )
    pd.DataFrame(fixed_rows).to_csv(
        output / "fixed_tests_and_ablation.csv", index=False
    )

    train_index, test_index = train_test_split(
        np.arange(len(dataframe)), test_size=0.2, random_state=SEED
    )
    model = models(FULL, SEED, quick)["Random Forest"]
    model.fit(dataframe.iloc[train_index][FULL], dataframe.iloc[train_index].absorbance)
    base_score = r2_score(
        dataframe.iloc[test_index].absorbance,
        model.predict(dataframe.iloc[test_index][FULL]),
    )
    groups = {
        "wavelength": ["wavelength"],
        "geometry_identity": ["geometry"],
        "primary_dimension": ["primary_dimension"],
        "secondary_dimension": ["secondary_dimension"],
        "dimension_types": ["primary_dimension_type", "secondary_dimension_type"],
    }
    importance_rows = []
    rng = np.random.default_rng(1000)
    for group, columns in groups.items():
        values = []
        for _ in range(10 if quick else 50):
            permuted = dataframe.iloc[test_index][FULL].copy()
            order = rng.permutation(len(permuted))
            permuted.loc[:, columns] = permuted[columns].to_numpy()[order]
            score = r2_score(
                dataframe.iloc[test_index].absorbance, model.predict(permuted)
            )
            values.append(base_score - score)
        importance_rows.append(
            {
                "feature_group": group,
                "mean_decrease_R2": np.mean(values),
                "sd_decrease_R2": np.std(values, ddof=1),
            }
        )

    importance = pd.DataFrame(importance_rows)
    importance.to_csv(output / "grouped_permutation_importance.csv", index=False)

    plot_data = summary[summary.protocol == "random_rows"].sort_values("MAE_mean")
    plt.figure(figsize=(8, 5))
    plt.barh(plot_data.model, plot_data.MAE_mean, xerr=plot_data.MAE_sd)
    plt.xlabel("Mean absolute error")
    plt.tight_layout()
    plt.savefig(output / "figures/model_comparison.png", dpi=250)
    plt.close()

    plot_data = pd.DataFrame(lodo_rows)
    plot_data = plot_data[plot_data.model == "Random Forest"].sort_values("MAE")
    plt.figure(figsize=(8, 6))
    plt.barh(plot_data.withheld_design, plot_data.MAE)
    plt.xlabel("LODO MAE")
    plt.tight_layout()
    plt.savefig(output / "figures/lodo_mae.png", dpi=250)
    plt.close()

    plot_data = importance.sort_values("mean_decrease_R2")
    plt.figure(figsize=(8, 5))
    plt.barh(
        plot_data.feature_group,
        plot_data.mean_decrease_R2,
        xerr=plot_data.sd_decrease_R2,
    )
    plt.xlabel("Decrease in R² after grouped permutation")
    plt.tight_layout()
    plt.savefig(output / "figures/grouped_importance.png", dpi=250)
    plt.close()

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "rows": len(dataframe),
        "designs": dataframe.design_id.nunique(),
        "wavelengths": dataframe.wavelength.nunique(),
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="hgcdte_absorbance.csv")
    parser.add_argument("--output", default="project1_outputs")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--quick", action="store_true")
    arguments = parser.parse_args()
    path = Path(arguments.data)

    if arguments.demo:
        path = Path("hgcdte_absorbance_demo.csv")
        make_demo(path)
        print(f"Synthetic demo data created: {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Use --demo or provide --data."
        )

    dataframe = load_data(path)
    run(dataframe, Path(arguments.output), arguments.quick)
    metadata_path = Path(arguments.output) / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_file"] = str(path)
    metadata["input_sha256"] = sha256(path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Outputs saved to {arguments.output}")


if __name__ == "__main__":
    main()
