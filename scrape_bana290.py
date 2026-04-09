import csv
import json
import re
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

URL = "https://bana290-assignment1.netlify.app/"
OUTPUT_CSV = "bana290_profiles.csv"
OUTPUT_JSON = "bana290_profiles.json"


def clean_text(value: str) -> str:
    return " ".join(value.split()).strip()


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
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)


def save_json(headers, rows, output_path):
    records = [dict(zip(headers, row)) for row in rows]
    with open(output_path, "w", encoding="utf-8") as jsonfile:
        json.dump(records, jsonfile, indent=2, ensure_ascii=False)


def cleanup_outputs():
    for path in (Path(OUTPUT_CSV), Path(OUTPUT_JSON)):
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


def main():
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
        print("No table data found on the page.")
        sys.exit(1)

    rows = [clean_row(headers, row) for row in rows]
    rows = drop_incomplete_rows(headers, rows)
    cleanup_outputs()
    save_csv(headers, rows, OUTPUT_CSV)
    save_json(headers, rows, OUTPUT_JSON)
    print(f"Saved {len(rows)} rows to {OUTPUT_CSV} and {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
