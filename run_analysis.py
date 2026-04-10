# End-to-end analysis script created from interactive prompts.
# Prompt: "combine the python files to be one end to end run from clean scraping to baseline analysis to propensity analysis. Have it still output to its respective files"
# Prompt: "Baseline Correlation: Run a naive OLS regression of Revenue Growth on AI Adoption and record the coefficient, save to an output file"
# Prompt: "Plot the distribution of propensity scores for both groups. Calculate the Standardized Mean Difference (SMD) for covariates before and after matching. Perform nearest–neighbor matching and re–estimate the effect of AI. - Save this output to a separate file"
# Prompt: "Can you include the prompts that created the code blocks as comments in the run analysis script?"

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://bana290-assignment1.netlify.app/"
OUTPUT_CSV = Path("bana290_profiles.csv")
OUTPUT_JSON = Path("bana290_profiles.json")
BASELINE_OUTPUT_JSON = Path("baseline_regression_output.json")
PSM_OUTPUT_JSON = Path("propensity_score_matching_output.json")
PSM_OUTPUT_PLOT = Path("propensity_score_distribution.png")
PSM_OUTPUT_SMD_PLOT = Path("propensity_score_smd_comparison.png")
PSM_OUTPUT_TEXT = Path("propensity_score_matching_output.txt")
PSM_CALIPER = 0.10


def clean_text(value: str) -> str:
    return " ".join(str(value).split()).strip()


def parse_html_table(table):
    headers = [clean_text(th.get_text()) for th in table.find_all("th")]
    if not headers:
        headers = [clean_text(td.get_text()) for td in table.find_all("tr")[0].find_all("td")]
    rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        row = [clean_text(cell.get_text()) for cell in cells]
        if len(row) == len(headers):
            rows.append(row)
    if headers and rows and rows[0] == headers:
        rows = rows[1:]
    return headers, rows


def parse_markdown_table(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 3:
        return [], []

    def split_row(line):
        return [clean_text(cell) for cell in line.strip("|").split("|")]

    header = split_row(table_lines[0])
    separator = split_row(table_lines[1])
    if not header or not separator:
        return [], []
    rows = [split_row(line) for line in table_lines[2:]]
    rows = [row for row in rows if len(row) == len(header)]
    return header, rows


def save_csv(headers, rows, output_path):
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)


def save_json(headers, rows, output_path):
    records = [dict(zip(headers, row)) for row in rows]
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def cleanup_outputs():
    for path in (OUTPUT_CSV, OUTPUT_JSON):
        if path.exists():
            path.unlink()


def is_missing_value(value: str) -> bool:
    if value is None:
        return True
    normalized = clean_text(str(value)).lower()
    return normalized in {"", "--", "n/a", "na", "unknown"}


def parse_numeric(value: str, allow_percent: bool = False):
    if value is None:
        return None

    text = clean_text(str(value)).lower()
    if is_missing_value(text):
        return None

    is_percent = "%" in text
    text = text.replace("$", "").replace("usd", "")
    text = text.replace(",", "").replace("%", "").replace("rev", "")
    text = text.replace("million", "m").replace("mn", "m")
    text = text.replace("billion", "b").replace("bn", "b")
    text = text.replace("k", "k")
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

    if "b" in text:
        value *= 1_000_000_000
    elif "m" in text:
        value *= 1_000_000
    elif "k" in text:
        value *= 1_000

    if is_percent and not allow_percent:
        return None
    return value


def normalize_binary(value: str):
    if value is None:
        return None
    normalized = clean_text(str(value)).lower().replace(",", "")
    if is_missing_value(normalized):
        return None

    if normalized in {"1", "yes", "y", "true", "ai enabled", "adopted", "production", "live", "in review", "pilot", "enabled"}:
        return 1
    if normalized in {"0", "no", "n", "false", "legacy only", "manual only", "not yet", "unknown", "n/a", "na", "--"}:
        return 0
    return None


def normalize_headers(headers):
    return [re.sub(r"[^a-z0-9]", "", clean_text(h).lower()) for h in headers]


def clean_row(headers, row):
    normalized_headers = normalize_headers(headers)
    normalized_row = [None] * len(row)

    for idx, value in enumerate(row):
        value = value if value is not None else ""
        header = normalized_headers[idx]

        if header == "annualrev":
            normalized_row[idx] = parse_numeric(value)
        elif header == "rdspend" or ("rd" in header and "spend" in header):
            normalized_row[idx] = parse_numeric(value)
        elif header in {"aiprogram", "aistatus", "aistatus", "ai"} or "ai" in header:
            normalized_row[idx] = normalize_binary(value)
        elif header == "revgrowth" or "growth" in header:
            normalized_row[idx] = parse_numeric(value, allow_percent=True)
        else:
            normalized_row[idx] = None if is_missing_value(value) else clean_text(str(value))

    return normalized_row


def drop_incomplete_rows(headers, rows):
    normalized_headers = normalize_headers(headers)
    required = {"revgrowth"}
    required_indices = [i for i, h in enumerate(normalized_headers) if h in required]
    if not required_indices:
        return rows

    filtered_rows = []
    for row in rows:
        if any(row[idx] is None for idx in required_indices):
            continue
        filtered_rows.append(row)
    return filtered_rows


# Scraping and cleaning section from the combine script prompt.
# This section downloads the site, extracts the directory table, standardizes missing values, and writes cleaned files.
def scrape_data() -> tuple[list[str], list[list[str]]]:
    response = requests.get(URL, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    headers = []
    rows = []

    if tables:
        for table in tables:
            hdrs, data = parse_html_table(table)
            if hdrs and data:
                headers, rows = hdrs, data
                break

    if not rows:
        headers, rows = parse_markdown_table(soup.get_text())

    if not headers or not rows:
        raise RuntimeError("No table data found on the page.")

    rows = [clean_row(headers, row) for row in rows]
    rows = drop_incomplete_rows(headers, rows)
    cleanup_outputs()
    save_csv(headers, rows, OUTPUT_CSV)
    save_json(headers, rows, OUTPUT_JSON)
    print(f"Saved {len(rows)} rows to {OUTPUT_CSV} and {OUTPUT_JSON}")
    return headers, rows


# Baseline OLS regression section from the baseline correlation prompt.
# This section prepares AI adoption and revenue growth and computes the naive coefficient.
def load_baseline_data(path: Path) -> pd.DataFrame:
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


def run_baseline_ols(df: pd.DataFrame) -> tuple[float, float]:
    x = df["ai_program"].to_numpy(dtype=float)
    y = df["rev_growth"].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return intercept, slope


def save_baseline_results(intercept: float, slope: float, n_obs: int, path: Path) -> None:
    result = {
        "n_observations": n_obs,
        "intercept": intercept,
        "ai_adoption_coefficient": slope,
        "model": "rev_growth ~ 1 + ai_program",
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def save_psm_text_summary(result: dict, path: Path) -> None:
    lines = [
        "Propensity Score Matching Analysis",
        "===============================\n",
        f"Caliper: {result['matching_caliper']}",
        f"Observations before matching: {result['n_observations_before']}",
        f"Observations matched: {result['n_matched']}",
        "\nBaseline OLS:\n",
        f"  Intercept: {result['baseline_ols']['intercept']}",
        f"  AI coefficient: {result['baseline_ols']['ai_adoption_coefficient']}",
        "\nMatched OLS:\n",
        f"  Intercept: {result['matched_ols']['intercept']}",
        f"  AI coefficient: {result['matched_ols']['ai_adoption_coefficient']}",
        "\nSMD before matching:",
    ]
    for cov, value in result['smd_before'].items():
        lines.append(f"  {cov}: {value:.4f}")
    lines.append("\nSMD after matching:")
    for cov, value in result['smd_after'].items():
        lines.append(f"  {cov}: {value:.4f}")
    lines.append("\nOutput files:")
    lines.append(f"  Propensity score plot: {result['propensity_score_plot']}")
    lines.append(f"  SMD comparison plot: {result['smd_comparison_plot']}")
    path.write_text("\n".join(lines), encoding="utf-8")


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


# Propensity score matching section from the PSM prompt.
# This section builds propensity scores, plots distributions, computes SMD before and after matching, and re-estimates AI effect.
def load_psm_data(path: Path) -> pd.DataFrame:
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
    X = df[["founded", "team_size", "annual_rev", "rd_spend", "digital_sales", "customer_accts"]].copy()
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


def nearest_neighbor_matching(df: pd.DataFrame, ps: np.ndarray, caliper: float | None = None) -> pd.DataFrame:
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
        nearest_pos = np.argmin(distances)
        nearest_idx = available_idx[nearest_pos]
        nearest_distance = distances[nearest_pos]
        if caliper is not None and nearest_distance > caliper:
            continue
        matches.append((treated_row.name, nearest_idx))
        available.remove(nearest_idx)

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


def plot_smd_comparison(smd_before: dict[str, float], smd_after: dict[str, float], path: Path) -> None:
    covariates = list(smd_before.keys())
    before_values = [smd_before[cov] for cov in covariates]
    after_values = [smd_after.get(cov, 0.0) for cov in covariates]

    x = np.arange(len(covariates))
    width = 0.35

    plt.figure(figsize=(max(12, len(covariates) * 0.25), 8))
    plt.bar(x - width / 2, before_values, width, label="Before", color="tab:blue", alpha=0.8)
    plt.bar(x + width / 2, after_values, width, label="After", color="tab:orange", alpha=0.8)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axhline(0.1, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-0.1, color="gray", linestyle="--", linewidth=0.8)
    plt.xticks(x, covariates, rotation=90, fontsize=8)
    plt.ylabel("Standardized Mean Difference")
    plt.title("Covariate Balance: SMD Before and After Matching")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    # Prompt: "combine the python files to be one end to end run from clean scraping to baseline analysis to propensity analysis. Have it still output to its respective files"
    # Run the full pipeline in one pass.

    # Prompt: "I want to scrape the data from https://bana290-assignment1.netlify.app/"
    # Scrape and clean the raw site data, then write cleaned outputs.
    scrape_data()

    # Prompt: "Baseline Correlation: Run a naive OLS regression of Revenue Growth on AI Adoption and record the coefficient, save to an output file"
    # Load the cleaned data, estimate the naive baseline OLS effect, and save the coefficient output.
    baseline_df = load_baseline_data(OUTPUT_CSV)
    baseline_df = baseline_df.dropna(subset=["ai_program", "rev_growth"])
    intercept, slope = run_baseline_ols(baseline_df)
    save_baseline_results(intercept, slope, len(baseline_df), BASELINE_OUTPUT_JSON)
    print(f"Saved baseline regression result to {BASELINE_OUTPUT_JSON}")

    # Prompt: "Plot the distribution of propensity scores for both groups. Calculate the Standardized Mean Difference (SMD) for covariates before and after matching. Perform nearest–neighbor matching and re–estimate the effect of AI. - Save this output to a separate file"
    # Load the cleaned data, build covariates, estimate propensity scores, match, compute SMDs, and save matching results.
    psm_df = load_psm_data(OUTPUT_CSV)
    X = build_covariate_matrix(psm_df)
    ps, _ = compute_propensity_scores(psm_df, X)
    psm_df["propensity_score"] = ps
    plot_propensity_distribution(psm_df, PSM_OUTPUT_PLOT)

    covariate_names = list(X.columns)
    smd_before = compute_smd(pd.concat([psm_df[["ai_program", "rev_growth"]], X], axis=1), covariate_names)
    matched_df = nearest_neighbor_matching(psm_df, ps, caliper=PSM_CALIPER)
    smd_after = compute_smd(pd.concat([matched_df[["ai_program", "rev_growth"]], X.loc[matched_df.index]], axis=1), covariate_names)

    intercept_before, slope_before = estimate_effect(psm_df)
    intercept_after, slope_after = estimate_effect(matched_df)

    result = {
        "n_observations_before": len(psm_df),
        "n_matched": len(matched_df),
        "matching_caliper": PSM_CALIPER,
        "propensity_score_plot": str(PSM_OUTPUT_PLOT),
        "smd_comparison_plot": str(PSM_OUTPUT_SMD_PLOT),
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
    PSM_OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot_smd_comparison(smd_before, smd_after, PSM_OUTPUT_SMD_PLOT)
    save_psm_text_summary(result, PSM_OUTPUT_TEXT)
    print(
        f"Saved PSM output to {PSM_OUTPUT_JSON}, text summary to {PSM_OUTPUT_TEXT},"
        f" plot to {PSM_OUTPUT_PLOT}, and SMD comparison plot to {PSM_OUTPUT_SMD_PLOT}"
    )


if __name__ == "__main__":
    main()
