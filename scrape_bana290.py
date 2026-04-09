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

    save_csv(headers, rows, OUTPUT_CSV)
    save_json(headers, rows, OUTPUT_JSON)
    print(f"Saved {len(rows)} rows to {OUTPUT_CSV} and {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
