import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INPUT_CSV = Path("bana290_profiles.csv")
OUTPUT_JSON = Path("propensity_score_matching_output.json")
OUTPUT_PLOT = Path("propensity_score_distribution.png")


def clean_text(value: str) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def parse_number(value: str):
    if value is None:
        return None
    text = clean_text(value).lower()
    if text in {"", "--", "n/a", "na", "unknown"}:
        return None

    text = text.replace("$", "").replace("usd", "")
    text = text.replace("%", "").replace("rev", "")
    text = text.replace(",", "")
    unit = 1.0
    if re.search(r"\b(billion|bn)\b", text):
        unit = 1_000_000_000
    elif re.search(r"\b(million|mn)\b", text):
        unit = 1_000_000
    elif re.search(r"k\b", text):
        unit = 1_000

    text = re.sub(r"\b(billion|million|bn|mn|k)\b", "", text)
    text = text.strip()

    if not text:
        return None

    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None

    try:
        value = float(match.group(0))
    except ValueError:
        return None

    return value * unit


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Rev Growth (YoY)": "rev_growth",
        "AI Program": "ai_program",
        "Annual Rev.": "annual_rev",
        "R&D Spend": "rd_spend",
        "Team Size": "team_size",
        "Digital Sales": "digital_sales",
        "Customer Accts": "customer_accts",
        "Founded": "founded",
        "Segment": "segment",
        "Cloud Stack": "cloud_stack",
        "Compliance Tier": "compliance_tier",
        "Fraud Exposure": "fraud_exposure",
        "Funding Stage": "funding_stage",
    })

    df["team_size"] = df["team_size"].astype(str).apply(parse_number)
    df["digital_sales"] = df["digital_sales"].astype(str).apply(parse_number)
    df["customer_accts"] = df["customer_accts"].astype(str).apply(parse_number)
    df["ai_program"] = pd.to_numeric(df["ai_program"], errors="coerce")
    df["rev_growth"] = pd.to_numeric(df["rev_growth"], errors="coerce")

    covariates = [
        "founded",
        "team_size",
        "annual_rev",
        "rd_spend",
        "digital_sales",
        "customer_accts",
        "segment",
        "cloud_stack",
        "compliance_tier",
        "fraud_exposure",
        "funding_stage",
    ]

    df = df[["ai_program", "rev_growth"] + covariates].copy()
    df = df.dropna(subset=["ai_program", "rev_growth"])
    df = df.dropna(subset=["team_size", "annual_rev", "digital_sales", "customer_accts"])
    df = df.dropna(subset=["segment", "cloud_stack", "compliance_tier", "fraud_exposure", "funding_stage"])
    return df


def build_covariate_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[
        ["founded", "team_size", "annual_rev", "rd_spend", "digital_sales", "customer_accts"]
    ].copy()
    cat_cols = ["segment", "cloud_stack", "compliance_tier", "fraud_exposure", "funding_stage"]
    dummies = pd.get_dummies(df[cat_cols], drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    X = X.fillna(0)
    return X


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-6):
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        linear = X @ beta
        p_hat = sigmoid(linear)
        weights = p_hat * (1 - p_hat)
        grad = X.T @ (y - p_hat)
        W = np.diag(weights)
        hess = -(X.T @ W @ X)
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hess) @ grad
        beta_new = beta - delta
        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def compute_propensity_scores(df: pd.DataFrame, X: pd.DataFrame) -> np.ndarray:
    y = df["ai_program"].astype(float).to_numpy()
    X_design = np.column_stack([np.ones(len(X)), X.to_numpy(dtype=float)])
    beta = fit_logistic_regression(X_design, y)
    ps = sigmoid(X_design @ beta)
    return ps, beta


def nearest_neighbor_matching(df: pd.DataFrame, ps: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["propensity_score"] = ps
    treated = df[df["ai_program"] == 1.0].copy()
    control = df[df["ai_program"] == 0.0].copy()
    matches = []
    available = set(control.index.tolist())

    for _, treated_row in treated.iterrows():
        if not available:
            break
        available_idx = np.array(list(available))
        distances = np.abs(control.loc[available_idx, "propensity_score"].to_numpy() - treated_row["propensity_score"])
        chosen_label = available_idx[np.argmin(distances)]
        matches.append((treated_row.name, chosen_label))
        available.remove(chosen_label)

    matched_idx = [idx for pair in matches for idx in pair]
    return df.loc[matched_idx].copy()


def smd_for_numeric(series_t, series_c):
    mean_t = series_t.mean()
    mean_c = series_c.mean()
    var_t = series_t.var(ddof=1)
    var_c = series_c.var(ddof=1)
    pooled_sd = np.sqrt((var_t + var_c) / 2)
    if pooled_sd == 0:
        return 0.0
    return (mean_t - mean_c) / pooled_sd


def compute_smd(df: pd.DataFrame, covariates: list[str]) -> dict[str, float]:
    treated = df[df["ai_program"] == 1.0]
    control = df[df["ai_program"] == 0.0]
    smd = {}
    for cov in covariates:
        if cov in df.columns:
            smd[cov] = smd_for_numeric(treated[cov], control[cov])
    return smd


def estimate_effect(df: pd.DataFrame) -> tuple[float, float]:
    y = df["rev_growth"].to_numpy(dtype=float)
    x = df["ai_program"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return intercept, slope


def plot_propensity_distribution(df: pd.DataFrame, path: Path) -> None:
    treated = df[df["ai_program"] == 1.0]["propensity_score"]
    control = df[df["ai_program"] == 0.0]["propensity_score"]
    plt.figure(figsize=(9, 6))
    plt.hist(
        [treated, control],
        bins=20,
        density=True,
        alpha=0.6,
        label=["AI=1", "AI=0"],
        color=["tab:blue", "tab:orange"],
    )
    plt.xlabel("Propensity Score")
    plt.ylabel("Density")
    plt.title("Propensity Score Distribution by AI Adoption")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    df = load_data(INPUT_CSV)
    X = build_covariate_matrix(df)
    ps, beta = compute_propensity_scores(df, X)
    df["propensity_score"] = ps

    plot_propensity_distribution(df, OUTPUT_PLOT)

    covariate_names = list(X.columns)
    smd_before = compute_smd(pd.concat([df[["ai_program", "rev_growth"]], X], axis=1), covariate_names)
    matched_df = nearest_neighbor_matching(df, ps)
    smd_after = compute_smd(pd.concat([matched_df[["ai_program", "rev_growth"]], X.loc[matched_df.index]], axis=1), covariate_names)

    intercept_before, slope_before = estimate_effect(df)
    intercept_after, slope_after = estimate_effect(matched_df)

    result = {
        "n_observations_before": len(df),
        "n_matched": len(matched_df),
        "propensity_score_plot": str(OUTPUT_PLOT),
        "baseline_ols": {
            "intercept": intercept_before,
            "ai_adoption_coefficient": slope_before,
        },
        "matched_ols": {
            "intercept": intercept_after,
            "ai_adoption_coefficient": slope_after,
        },
        "smd_before": smd_before,
        "smd_after": smd_after,
    }
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved PSM output to {OUTPUT_JSON} and plot to {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
