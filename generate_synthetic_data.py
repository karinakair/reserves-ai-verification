"""
Generate SYNTHETIC reserves data that mimics real-world PRMS reporting.

All field names and numbers are fictional. No corporate data is used.

The script produces three files in three different layouts on purpose,
because real subsidiaries report in inconsistent formats:
  format A  - wide Excel table (one row per field, categories as columns)
  format B  - long Excel table (one row per field x category)
  format C  - CSV in cp1251 encoding with Russian headers and a title block
"""
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

COMPANIES = {
    "Company_A": ["Field_A1", "Field_A2", "Field_A3", "Field_A4", "Field_A5"],
    "Company_B": ["Field_B1", "Field_B2", "Field_B3", "Field_B4"],
    "Company_C": ["Field_C1", "Field_C2", "Field_C3", "Field_C4", "Field_C5", "Field_C6"],
}
CATEGORIES = ["1P", "2P", "3P"]


def make_base() -> pd.DataFrame:
    """Company (internal) and auditor estimates for YE2024 and YE2025, in MMbbl."""
    rows = []
    for company, fields in COMPANIES.items():
        for field in fields:
            p1_prev = RNG.uniform(20, 400)
            production = p1_prev * RNG.uniform(0.04, 0.09)          # annual production
            additions = production * RNG.uniform(0.5, 1.3)          # reserve additions
            p1 = p1_prev - production + additions
            ratios = {"1P": 1.0, "2P": RNG.uniform(1.3, 1.7), "3P": RNG.uniform(1.8, 2.4)}
            for cat in CATEGORIES:
                company_est = p1 * ratios[cat]
                auditor_est = company_est * RNG.normal(0.97, 0.03)  # auditor is usually a bit lower
                rows.append(dict(company=company, field=field, category=cat,
                                 ye2024=p1_prev * ratios[cat], ye2025_company=company_est,
                                 ye2025_auditor=auditor_est, production_2025=production))
    df = pd.DataFrame(rows)

    # Inject realistic anomalies the pipeline should find
    def bump(field, cat, col, k):
        m = (df.field == field) & (df.category == cat)
        df.loc[m, col] *= k

    bump("Field_A3", "1P", "ye2025_auditor", 0.62)   # auditor cut 1P by ~38%
    bump("Field_B2", "2P", "ye2025_company", 1.45)   # company overestimates 2P
    bump("Field_C5", "3P", "ye2025_auditor", 0.55)   # large 3P downgrade
    bump("Field_C2", "1P", "ye2025_company", 1.30)   # 1P jump without matching production history
    return df.round(2)


def write_format_a(df):
    """Wide layout, Company_A."""
    d = df[df.company == "Company_A"]
    wide = d.pivot_table(index="field", columns="category",
                         values=["ye2024", "ye2025_company", "ye2025_auditor"])
    wide.columns = [f"{v}_{c}" for v, c in wide.columns]
    wide["production_2025"] = d.groupby("field").production_2025.first()
    wide.reset_index().to_excel(RAW / "Company_A_reserves_wide.xlsx", index=False)


def write_format_b(df):
    """Long layout, Company_B, different column names."""
    d = df[df.company == "Company_B"].rename(columns={
        "field": "Field Name", "category": "Reserve Class", "ye2024": "Prev YE",
        "ye2025_company": "Internal Estimate", "ye2025_auditor": "Auditor Estimate",
        "production_2025": "Annual Production"})
    d.drop(columns="company").to_excel(RAW / "Company_B_reserves_long.xlsx",
                                       index=False, sheet_name="Reserves")


def write_format_c(df):
    """CSV in cp1251 with Russian headers, a title block and comma decimals."""
    d = df[df.company == "Company_C"].rename(columns={
        "field": "Месторождение", "category": "Категория", "ye2024": "Запасы на 01.01.2025",
        "ye2025_company": "Оценка компании", "ye2025_auditor": "Оценка аудитора",
        "production_2025": "Добыча за год"}).drop(columns="company")
    path = RAW / "Company_C_reserves.csv"
    with open(path, "w", encoding="cp1251", newline="") as f:
        f.write("Отчет о запасах (синтетические данные)\n")
        f.write("Единицы: млн барр.\n\n")
        d.to_csv(f, sep=";", decimal=",", index=False)


if __name__ == "__main__":
    base = make_base()
    write_format_a(base)
    write_format_b(base)
    write_format_c(base)
    print(f"Synthetic data written to {RAW}")
