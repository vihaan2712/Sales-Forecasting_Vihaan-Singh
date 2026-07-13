# Sales Forecasting & Demand Intelligence System
**Vihaan Singh — Internship Final Project — Submitted 13 July 2026**
#Live Dashboard
 **Streamlit App:** [Open the Live Sales Forecasting Dashboard](https://sales-forecastingvihaan-singh-a4gxfzvb6fyl59bjra9idx.streamlit.app/)

## What's in this folder

| File / Folder | What it is |
|---|---|
| `analysis.ipynb` | Full notebook, Tasks 1–6, already executed end-to-end (every chart and number was produced by running this code, not typed in). Open it and you'll see all outputs without needing to re-run anything. |
| `train.csv` | Superstore Sales dataset (primary data source, all 8 tasks). |
| `data/vgsales.csv` | Video Game Sales dataset (supplementary, used only in Task 5's multi-source merge exercise). |
| `charts/` | All chart PNGs, exported from the notebook. |
| `app.py` | Task 7 — the Streamlit dashboard (4 pages: Sales Overview, Forecast Explorer, Anomaly Report, Product Demand Segments). |
| `summary.docx` | Task 8 — the 2-page executive report, written for a non-technical Head of Supply Chain / CFO audience. |
| `requirements.txt` | Lean dependency list for **running/deploying `app.py`** (Streamlit Cloud). |
| `requirements-notebook.txt` | Full dependency list for **re-executing `analysis.ipynb`** (includes Prophet, XGBoost, etc. not needed by the dashboard). |

## Running things yourself

**Dashboard (local):**
```bash
pip install -r requirements.txt
streamlit run app.py
```

**Dashboard (Streamlit Community Cloud):** push this folder to a GitHub repo and point
[share.streamlit.io](https://share.streamlit.io) at `app.py`. `requirements.txt` was kept deliberately free of
Prophet/XGBoost (not used by the dashboard) so the cloud build stays fast and doesn't need Prophet's `cmdstan`
compile step.

**Notebook:**
```bash
pip install -r requirements-notebook.txt
jupyter notebook analysis.ipynb
```

## A note on how this was built

Every result in the notebook, dashboard, and report comes from actually running the code against the real datasets —
including the messy parts: a SARIMA grid search that occasionally needed a business floor at zero, a cluster-labeling
rule that got a re-check and a fix partway through Task 6, and growth comparisons that were deliberately restructured
(year-over-year instead of month-over-month) once an early version gave a misleading "everything is declining" signal
purely from comparing against December's seasonal peak. These are documented honestly in the notebook rather than
smoothed over, in the same spirit as the assignment's own grading note.
