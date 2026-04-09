import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CSV = Path("bana290_profiles.csv")
OUTPUT_JSON = Path("baseline_regression_output.json")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Rev Growth (YoY)": "rev_growth",
        "AI Program": "ai_program",
    })

    df["ai_program"] = (
        df["ai_program"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["ai_program"] = pd.to_numeric(df["ai_program"], errors="coerce")

    df["rev_growth"] = (
        df["rev_growth"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("rev", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["rev_growth"] = pd.to_numeric(df["rev_growth"], errors="coerce")

    return df[["ai_program", "rev_growth"]]


def run_ols(df: pd.DataFrame) -> tuple[float, float]:
    x = df["ai_program"].to_numpy(dtype=float)
    y = df["rev_growth"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return intercept, slope


def save_results(intercept: float, slope: float, n_obs: int, path: Path) -> None:
    result = {
        "n_observations": n_obs,
        "intercept": intercept,
        "ai_adoption_coefficient": slope,
        "model": "rev_growth ~ 1 + ai_program",
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main() -> None:
    df = load_data(INPUT_CSV)
    df = df.dropna(subset=["ai_program", "rev_growth"])
    intercept, slope = run_ols(df)
    save_results(intercept, slope, len(df), OUTPUT_JSON)
    print(f"Saved baseline regression result to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
