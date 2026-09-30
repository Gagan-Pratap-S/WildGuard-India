# 🐾 WildGuard India — Wildlife Conservation Analytics & Risk Dashboard

## Overview

**WildGuard India** is a data-driven conservation decision-support dashboard that analyses wildlife population trends in India using the Living Planet Index (LPI) 2024 public dataset. The dashboard transforms raw population time-series data into actionable conservation insights, helping decision-makers understand which species and ecosystems require the most urgent attention.

## Problem Statement

India is one of the world's most biodiverse countries, yet faces accelerating threats from habitat loss, pollution, climate change, and human-wildlife conflict. Effective conservation requires evidence-based prioritisation — understanding which populations are declining, where, and how fast.

**Central question:** *What do India's wildlife population trends tell us about conservation risk, and where should conservation attention be prioritised?*

## Objectives

1. **Clean and prepare** the India subset of the LPI dataset for analysis.
2. **Identify population trends** across species, taxonomic groups, and ecosystems.
3. **Compute meaningful KPIs** to quantify conservation status at a glance.
4. **Assess conservation risk** by identifying the most rapidly declining populations.
5. **Generate actionable insights** following the FACT → INSIGHT → RISK → ACTION framework.
6. **Present findings** in a clean, interactive Streamlit dashboard accessible to non-technical decision-makers.

## Dataset

This project uses the **India subset** of the [Living Planet Index 2024 Public Data](https://www.livingplanetindex.org/data_portal).

- **Source:** Living Planet Index (LPI), managed by the Zoological Society of London (ZSL) and WWF.
- **Dataset file:** `LPD_2024_public.csv`
- **Scope:** All population time-series with `Country == "India"` and `Included in LPR2024 == 1` (the quality-checked, non-replicate series used in the official Living Planet Report 2024).

### Key dataset characteristics (India subset):

| Metric | Value |
|--------|-------|
| Population series | 280 (post-filtered from 289 LPR2024 series) |
| Unique species | ~128 |
| Taxonomic classes | 5 (Aves, Mammalia, Reptilia, Actinopteri, Elasmobranchii) |
| Ecosystems | 3 (Terrestrial, Freshwater, Marine) |
| Observation period | 1956–2018 |
| Median observations per series | ~3 |

## Data Processing

### Filtering
1. **Country filter:** Exact match on `Country == "India"` (309 raw series; excludes "British Indian Ocean Territory").
2. **LPI inclusion filter:** Only series with `Included in LPR2024 == 1` are retained (289 series) — these are the non-replicate, quality-checked series recommended by the LPI team.
3. **Minimum data requirement:** Series with fewer than 2 non-null, positive population values are excluded (9 series removed, leaving **280 final analyzed series** for trend computation).

### Cleaning
- `NULL` string values replaced with proper `NaN`.
- Year columns converted to numeric; non-numeric entries coerced to `NaN`.
- Categorical fields stripped of whitespace inconsistencies.
- Empty year columns (no India data) dropped.
- Trailing unnamed columns removed.
- No data was randomly sampled — all qualifying India records are preserved.

## Methodology

### Population Trend Calculation

LPI population series use heterogeneous abundance measures (counts, density, indices, encounter rates) that **cannot be meaningfully summed** across species or series.

Instead, trends are computed using **log-linear regression**:

1. For each population series, we regress `ln(population value)` on `year`.
2. The **slope** of this regression gives the mean proportional rate of change per year.
3. **Annualised % change** = `(e^slope − 1) × 100`.
4. **Total % change** = `(e^(slope × span) − 1) × 100`, where `span` is the number of years between first and last observation.

This approach:
- Handles different measurement units correctly (relative change is unit-free).
- Provides a consistent, comparable metric across all series.
- Is aligned with the LPI methodology that uses geometric means of annual rates of change.

### Trend Classification

Each series is classified based on the slope and p-value:
- **Declining:** slope < −0.01 and p < 0.1
- **Increasing:** slope > +0.01 and p < 0.1
- **Stable:** |slope| ≤ 0.01 and p < 0.1
- **Uncertain:** p ≥ 0.1 (insufficient statistical evidence)

### Aggregate Index

An LPI-style aggregate index is computed by:
1. Computing annual lambda values: `λ = ln(N_t / N_{t−1})` for each consecutive observation pair within each series.
2. Averaging λ across all series for each year.
3. Cumulating: `Index_t = Index_{t−1} × e^(mean_λ_t)`, starting from a base of 100.

### Risk Identification

High-risk populations are identified as those with:
- Trend classification = "Declining"
- Annual % change < −3% per year

## Dashboard

The dashboard is organised into three pages:

### Page 1 — Executive Overview
- Key KPI cards (series count, species count, declining count, % declining, median annual change, taxonomic class count)
- Aggregate wildlife population index chart
- Trend classification pie chart (declining/increasing/stable/uncertain)
- Automatically derived insights

### Page 2 — Wildlife Trends
- Trend distribution by taxonomic class (grouped bar chart)
- Annual % change distribution (histogram)
- Annual % change by ecosystem (box plot)
- Top 15 most declining population series
- Top 15 most increasing population series
- Species coverage by class

### Page 3 — Conservation Risk & Insights
- High-risk population series table
- Risk indicators by taxonomic class and ecosystem
- Geographic map of monitored populations (colour-coded by trend)
- 5 data-driven insights following FACT → INSIGHT → RISK/OPPORTUNITY → ACTION format
- Recommended conservation actions
- Honest limitations statement

## Key Insights

*(Derived from the actual processed India LPI data)*

1. **Bird species dominate monitoring effort:** Aves represents the largest share of monitored population series, potentially masking under-monitored groups.
2. **Freshwater ecosystems are heavily represented** in the India LPI data, reflecting the importance of wetland bird counts and river surveys.
3. **A substantial proportion of series show declining trends**, indicating broad population pressure across India's wildlife.
4. **Several mammal populations show strong declines**, particularly large mammals in terrestrial ecosystems.
5. **Conservation success stories exist** — some populations show clear increasing trends, suggesting that targeted conservation interventions can work.
6. **Data sparsity is a major limitation** — the median number of observations per series is very low, limiting trend reliability for many populations.

## Recommended Actions

1. Expand long-term population monitoring for under-surveyed taxonomic groups.
2. Prioritise habitat protection in regions with the highest concentration of declining species.
3. Develop species-specific recovery plans for populations with the steepest declines.
4. Study and replicate conservation strategies associated with increasing population trends.
5. Re-survey populations with data gaps exceeding 10 years to update baseline data.

## Tech Stack

| Component | Technology |
|-----------|------------|
| Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualisation | Plotly |
| Statistics | SciPy |
| Language | Python 3.10+ |

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run wildguard.py
```

The dashboard will open in your default browser at `http://localhost:8501`.

## Project Structure

```
WildGuard-India/
├── wildguard.py          # Main application (data + analysis + dashboard)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── project_report.pdf    # Detailed project report
└── LPD_2024_public.csv   # LPI 2024 public dataset
```

## Limitations

- **Observational data:** All trends are based on observational population estimates. Correlations and patterns do not imply causation.
- **Sampling bias:** Well-studied taxa (birds, large mammals) are over-represented; invertebrates, amphibians, and plants are absent or under-represented.
- **Measurement heterogeneity:** Different series use different abundance measures (counts, density, encounter rates). Log-linear trend analysis handles this via relative change, but comparisons of absolute values across series are not meaningful.
- **Temporal sparsity:** Many India series have very few data points (2–3 observations), which limits the precision and reliability of trend estimates.
- **Geographic coverage:** The dataset does not uniformly cover all of India's biodiversity-rich regions.
- **No threat data:** The LPI dataset does not include direct information about threats or drivers; associations between trends and possible causes are inferred, not proven.
- **Historical bias:** Some series have their most recent observations before 2000, meaning current population status may differ from the trend computed here.

---

**Data Source:** [Living Planet Index — Data Portal](https://www.livingplanetindex.org/data_portal)  
**Citation:** LPI 2024. Living Planet Index database. 2024. www.livingplanetindex.org
