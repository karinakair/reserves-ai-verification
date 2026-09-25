"""
PRMS reserves verification pipeline.

1. Extract   - read heterogeneous subsidiary reports (3 layouts, 2 encodings)
2. Normalize - map everything to one tidy schema
3. KPIs      - Reserve Replacement Ratio (RRR) and Reserve Life Index (RLI)
4. Detect    - flag suspicious company vs auditor discrepancies
               (rule-based thresholds + Isolation Forest)
5. Explain   - generate a short natural-language note per anomaly
               (template by default, LLM if ANTHROPIC_API_KEY is set)
6. Report    - Excel workbook + charts ready for Power BI / management review
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "output"
OUT.mkdir(exist_ok=True)

SCHEMA = ["company", "field", "category", "ye2024", "ye2025_company",
          "ye2025_auditor", "production_2025"]
DIFF_THRESHOLD = 0.15   # >15% company vs auditor gap is worth a look


# ---------------------------------------------------------------- 1-2. extract + normalize
def read_format_a(path):
    wide = pd.read_excel(path)
    long = wide.melt(id_vars=["field", "production_2025"], var_name="metric")
    long[["metric", "category"]] = long.metric.str.rsplit("_", n=1, expand=True)
    tidy = long.pivot_table(index=["field", "category", "production_2025"],
                            columns="metric", values="value").reset_index()
    tidy["company"] = "Company_A"
    return tidy[SCHEMA]


def read_format_b(path):
    df = pd.read_excel(path, sheet_name="Reserves").rename(columns={
        "Field Name": "field", "Reserve Class": "category", "Prev YE": "ye2024",
        "Internal Estimate": "ye2025_company", "Auditor Estimate": "ye2025_auditor",
        "Annual Production": "production_2025"})
    df["company"] = "Company_B"
    return df[SCHEMA]


def read_format_c(path):
    # find the real header row instead of hard-coding it
    with open(path, encoding="cp1251") as f:
        header_row = next(i for i, line in enumerate(f) if line.startswith("Месторождение"))
    df = pd.read_csv(path, sep=";", decimal=",", encoding="cp1251", skiprows=header_row)
    df = df.rename(columns={
        "Месторождение": "field", "Категория": "category", "Запасы на 01.01.2025": "ye2024",
        "Оценка компании": "ye2025_company", "Оценка аудитора": "ye2025_auditor",
        "Добыча за год": "production_2025"})
    df["company"] = "Company_C"
    return df[SCHEMA]


READERS = {"*_wide.xlsx": read_format_a, "*_long.xlsx": read_format_b, "*.csv": read_format_c}


def extract_all():
    frames = []
    for pattern, reader in READERS.items():
        for path in sorted(RAW.glob(pattern)):
            frames.append(reader(path))
            print(f"  read {path.name}")
    df = pd.concat(frames, ignore_index=True)
    assert df[SCHEMA].notna().all().all(), "missing values after normalization"
    return df


# ---------------------------------------------------------------- 3. KPIs
def add_kpis(df):
    df = df.copy()
    df["diff_pct"] = (df.ye2025_company - df.ye2025_auditor) / df.ye2025_auditor
    df["yoy_pct"] = (df.ye2025_company - df.ye2024) / df.ye2024
    for src in ["company", "auditor"]:
        additions = df[f"ye2025_{src}"] - df.ye2024 + df.production_2025
        df[f"rrr_{src}"] = additions / df.production_2025          # RRR, share of production replaced
        df[f"rli_{src}"] = df[f"ye2025_{src}"] / df.production_2025  # RLI, years
    return df


# ---------------------------------------------------------------- 4. detect
def detect_anomalies(df):
    df = df.copy()
    features = df[["diff_pct", "yoy_pct", "rrr_company"]]
    X = StandardScaler().fit_transform(features)
    model = IsolationForest(n_estimators=300, contamination=0.1, random_state=0)
    df["iforest_flag"] = model.fit_predict(X) == -1
    df["anomaly_score"] = -model.score_samples(X)
    df["rule_flag"] = df.diff_pct.abs() > DIFF_THRESHOLD
    df["is_anomaly"] = df.iforest_flag | df.rule_flag
    return df


# ---------------------------------------------------------------- 5. explain
def template_explanation(r):
    direction = "higher" if r.diff_pct > 0 else "lower"
    parts = [f"{r.field} ({r.company}), {r.category}: company estimate is "
             f"{abs(r.diff_pct):.0%} {direction} than the auditor's."]
    if abs(r.yoy_pct) > 0.2:
        parts.append(f"Year-over-year change of {r.yoy_pct:+.0%} is unusual.")
    if r.category == "1P" and r.rrr_company > 1.5:
        parts.append(f"Implied RRR of {r.rrr_company:.1f} needs support from new wells or data.")
    parts.append("Check: well tests, decline curve assumptions, category transfers.")
    return " ".join(parts)


def llm_explanation(r):
    """Optional: ask an LLM for a reviewer-style note. Falls back to the template."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = (
            "You are a reserves engineer reviewing an SPE PRMS audit. In 2 sentences, "
            "explain the likely technical reasons for this discrepancy and what to check.\n"
            f"{r[SCHEMA + ['diff_pct', 'yoy_pct', 'rrr_company']].to_dict()}")
        msg = client.messages.create(model=os.getenv("LLM_MODEL", "claude-sonnet-4-5"),
                                     max_tokens=200,
                                     messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception as e:  # no key, no package, no network
        print(f"  LLM unavailable ({type(e).__name__}), using template")
        return template_explanation(r)


def explain(df):
    use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    flagged = df[df.is_anomaly].sort_values("anomaly_score", ascending=False).copy()
    fn = llm_explanation if use_llm else template_explanation
    flagged["explanation"] = [fn(r) for _, r in flagged.iterrows()]
    return flagged


# ---------------------------------------------------------------- 6. report
def plot(df):
    p2 = df[df.category == "2P"].sort_values("ye2025_company")
    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(p2))
    ax.barh(y - 0.2, p2.ye2025_company, 0.4, label="Company", color="#2b6cb0")
    ax.barh(y + 0.2, p2.ye2025_auditor, 0.4, label="Auditor", color="#a0aec0")
    ax.set_yticks(y, p2.field)
    for i, flag in enumerate(p2.is_anomaly):
        if flag:
            ax.get_yticklabels()[i].set_color("#c53030")
    ax.set_xlabel("2P reserves, MMbbl (synthetic)")
    ax.set_title("2P reserves: company vs auditor (red = flagged)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "2p_company_vs_auditor.png", dpi=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    ok, bad = df[~df.is_anomaly], df[df.is_anomaly]
    ax.scatter(ok.yoy_pct * 100, ok.diff_pct * 100, c="#a0aec0", label="normal")
    ax.scatter(bad.yoy_pct * 100, bad.diff_pct * 100, c="#c53030", label="flagged")
    for r in bad.itertuples():
        ax.annotate(f"{r.field} {r.category}", (r.yoy_pct * 100, r.diff_pct * 100),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.axhspan(-DIFF_THRESHOLD * 100, DIFF_THRESHOLD * 100, color="#edf2f7", zorder=0)
    ax.set_xlabel("Year-over-year change, %")
    ax.set_ylabel("Company vs auditor gap, %")
    ax.set_title("Anomaly map")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "anomaly_map.png", dpi=150)


def main():
    print("1-2. Extract + normalize")
    df = extract_all()
    print(f"  {len(df)} rows, {df.field.nunique()} fields, {df.company.nunique()} companies")
    print("3. KPIs")
    df = add_kpis(df)
    print("4. Anomaly detection")
    df = detect_anomalies(df)
    print(f"  flagged {df.is_anomaly.sum()} of {len(df)} records")
    print("5. Explanations")
    flagged = explain(df)
    print("6. Report")
    summary = (df[df.category == "1P"].groupby("company")
               [["ye2025_company", "ye2025_auditor", "production_2025"]].sum())
    summary["rrr_company"] = ((summary.ye2025_company - df[df.category == "1P"]
                               .groupby("company").ye2024.sum() + summary.production_2025)
                              / summary.production_2025)
    summary["rli_company_years"] = summary.ye2025_company / summary.production_2025
    with pd.ExcelWriter(OUT / "reserves_verification_report.xlsx") as xw:
        summary.round(2).to_excel(xw, sheet_name="Summary_1P")
        flagged[["company", "field", "category", "diff_pct", "yoy_pct", "rrr_company",
                 "anomaly_score", "explanation"]].round(3).to_excel(
            xw, sheet_name="Flagged", index=False)
        df.round(3).to_excel(xw, sheet_name="All_data", index=False)
    df.to_csv(OUT / "reserves_unified.csv", index=False)  # Power BI ready
    plot(df)
    print("\nTop findings:")
    for e in flagged.explanation.head(5):
        print("  -", e)
    print(f"\nDone. Outputs in {OUT}")


if __name__ == "__main__":
    main()
