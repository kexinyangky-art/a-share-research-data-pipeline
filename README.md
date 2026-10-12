# A-Share Research Data Pipeline & Factor Exploration

An end-to-end research data project for A-share equities, covering **multi-source data integration, point-in-time alignment, data quality controls, factor engineering, exposure decomposition, and alternative-data exploration**.

The project is built on a research universe of **80 CSI 300 constituents** and is designed to demonstrate how raw market and fundamental data can be transformed into a reliable, research-ready dataset for quantitative analysis.

> **Scope:** research data / factor exploration  
> **Not intended as:** a production trading strategy or a claim of stable alpha

---

## Key Highlights

- Built a daily research panel with **88,648 stock-date observations** from 2022-01-04 to 2026-08-31.
- Integrated daily prices, industry metadata, quarterly fundamentals, valuation data, and institutional-research alternative data.
- Implemented **point-in-time (PIT)** joins using publication / announcement dates to prevent look-ahead bias.
- Added data-quality checks for duplicates, missing periods, field applicability, vendor inconsistencies, and source-history limitations.
- Engineered market and fundamental factors including **Size, Value, Momentum, Reversal, Volatility, Liquidity, Quality, Growth, and Leverage**.
- Applied daily cross-sectional **winsorization, z-scoring, industry neutralization, and factor residualization**.
- Evaluated factors using **Rank IC, ICIR, positive IC ratio, and top-minus-bottom quintile spreads**.
- Identified that the apparent liquidity signal was largely driven by **Size exposure**.
- Built institutional-attention features from alternative data and evaluated whether they contained information beyond traditional factors.

---

## Research Pipeline

```text
Universe
   ↓
Daily Market Data
   ↓
Company / Industry Metadata
   ↓
Quarterly Fundamentals
   ↓
Data Quality Audit & Cleaning
   ↓
Point-in-Time Research Panel
   ↓
Raw Factor Engineering
   ↓
Winsorization / Z-score / Industry Neutralization
   ↓
Factor Evaluation
   ↓
Historical Valuation Data
   ↓
Size / Value / Factor Exposure Decomposition
   ↓
Institutional Research Alternative Data
   ↓
Alternative Feature Engineering
   ↓
Exploratory Incremental-Information Analysis
```

---

## Dataset Overview

| Dataset | Coverage |
|---|---:|
| Research universe | 80 stocks |
| Daily market panel | 88,648 rows |
| Market-data period | 2022-01-04 to 2026-08-31 |
| Quarterly fundamental records | 1,425 rows |
| Fundamental period | 2022Q1 to 2026Q2 |
| Institutional-research events | 772 events |
| Stocks with institutional-research events | 79 |
| Mature alternative-data period | 2026-01-09 to 2026-08-31 |

The universe uses a subset of **current CSI 300 constituents**. Historical constituent membership is not reconstructed, so the project contains **survivorship bias** and should be interpreted as factor exploration rather than a production-grade index backtest.

---

## Point-in-Time Data Design

A core objective of the project is to ensure that no information is used before it becomes publicly available.

### Fundamental data

Each quarterly financial record contains:

- `report_date`: accounting period end
- `available_date`: date on which the required financial information became publicly available

Daily observations use the latest record satisfying:

```text
available_date <= trading_date
```

This prevents financial information from being backfilled into dates when investors could not yet have known it.

For example, if a quarterly report becomes available on Saturday, the new information first enters the research panel on the next trading day.

### Alternative data

Institutional-research events are aligned using **announcement date**, not merely the underlying research-event date.

This ensures that a research event only affects features once the information is public.

---

## Data Quality Framework

The project treats data quality as part of the research process rather than as a separate cleanup step.

### Coverage and uniqueness

Checks include:

- duplicate `code-date` records
- duplicate `code-report_date` records
- expected vs actual quarterly coverage
- stock-level coverage
- listing-date-aware missing-period checks

### Field applicability

Missing data is not automatically treated as bad data.

For example, gross margin is not a universally meaningful metric for financial institutions. The pipeline therefore distinguishes between:

- true missing / source errors
- structurally inapplicable fields
- vendor coverage limitations

### Vendor-field validation

The raw debt-to-asset ratio exhibited implausible jumps in many stock-quarter records.

A derived ratio was reconstructed using:

```text
Debt / Assets = 1 - 1 / Asset-Equity Multiplier
```

The pipeline:

1. preserves the raw vendor value,
2. calculates the derived accounting-consistent value,
3. flags large discrepancies,
4. uses the derived value when valid.

### Structural source limitations

The revenue field showed a recurring vendor pattern:

- Q1 / Q3: largely unavailable
- Q2 / Q4: available

Because this was a source-coverage issue rather than an economic signal, revenue was retained for traceability but excluded from the core research features.

---

## Factor Engineering

### Market Factors

**Momentum**

```text
20D Momentum = P_t / P_(t-20) - 1
```

**Short-Term Reversal**

```text
5D Reversal = - 5D Return
```

**Volatility**

20-day rolling standard deviation of daily returns, annualized using `sqrt(252)`.

**Liquidity**

Amihud-style illiquidity:

```text
abs(Return) / Trading Amount
```

The sign is reversed in the final factor so that:

```text
higher factor = more liquid
```

### Fundamental Factors

- **Quality:** ROE proxy
- **Growth:** Net Profit YoY
- **Low Leverage:** negative Debt-to-Asset Ratio

### Size

A historical free-float market-cap proxy is reconstructed from daily market data:

```text
Implied Float Shares
= Volume / (Turnover / 100)

Float Market Cap Proxy
= Unadjusted Close × Implied Float Shares

Size
= log(Float Market Cap Proxy)
```

A 20-day rolling median is used to stabilize implied float shares against vendor rounding noise.

### Value

Book-to-price proxy:

```text
Value = 1 / PB
```

Higher values represent cheaper stocks on a book-value basis.

---

## Cross-Sectional Processing

Each factor is processed daily using:

```text
Raw Factor
   ↓
1% / 99% Winsorization
   ↓
Cross-Sectional Z-score
   ↓
Industry Neutralization
   ↓
Re-standardization
   ↓
Research-Ready Factor
```

Processed factors satisfy approximately:

```text
Daily Mean ≈ 0
Daily Standard Deviation ≈ 1
```

Industry neutralization is implemented through within-industry demeaning for groups with sufficient observations.

---

## Factor Evaluation

Factors are evaluated at **month-end cross sections** to reduce the dependence caused by heavily overlapping forward-return windows.

Main evaluation metrics:

- Spearman Rank IC
- Median Rank IC
- Positive IC Ratio
- ICIR
- Top-minus-Bottom Quintile Return Spread

Forward returns are used only as **labels**, never as input features.

---

## Selected Research Findings

### 1. Short-Term Reversal

Short-term reversal was one of the more consistent characteristics in the selected universe.

| Horizon | Mean Rank IC | Positive IC Ratio | Mean Top-Bottom Spread |
|---|---:|---:|---:|
| 5D | **+0.0726** | 69.1% | +1.00% |
| 20D | **+0.0427** | 58.2% | +1.45% |

This suggests that stocks with weaker recent 5-day performance tended to perform better over the next short horizon in this sample.

---

### 2. Value

Book-to-price showed positive cross-sectional Rank IC:

| Horizon | Mean Rank IC |
|---|---:|
| 5D | **+0.0549** |
| 20D | **+0.0372** |

The result is exploratory and is not interpreted as a validated standalone Value strategy.

---

### 3. Liquidity Was Largely a Size Exposure

The original liquidity factor appeared predictive:

```text
20D Liquidity Rank IC = -0.0737
```

However, Liquidity and Size were strongly correlated:

```text
Average Daily Pearson Corr  = +0.6001
Average Daily Spearman Corr = +0.6892
```

Liquidity was therefore residualized cross-sectionally on Size:

```text
Liquidity = alpha + beta × Size + residual
```

After removing Size exposure:

```text
Size-Neutral Liquidity 20D Rank IC = -0.0168
```

while:

```text
Size 20D Rank IC = -0.0815
```

The daily Pearson correlation between the Size-neutral Liquidity factor and Size became approximately zero.

**Interpretation:** much of the apparent liquidity effect in this sample was actually associated with small-size exposure rather than an independent liquidity effect.

---

## Alternative Data: Institutional Research Attention

Institutional research / investor-relations activity was used as an alternative data source.

### Source-history limitation

Although the source was queried from 2022 onward, the returned history only covered:

```text
2025-10-11 to 2026-09-30
```

This creates a **left-censoring problem**.

Dates before 2025-10-11 cannot be interpreted as zero institutional attention because the earlier history is simply unavailable.

### 90-Day Warm-Up

Because several features use 90-day lookback windows, they are not considered fully observable until:

```text
2025-10-11 + 90 days
= 2026-01-09
```

The mature alternative-data period therefore becomes:

```text
2026-01-09 to 2026-08-31
```

This produces:

- 8 mature month-end snapshots
- 7 usable month-end forward-return observations

The final month cannot be evaluated because the market panel ends on 2026-08-31 and therefore has no subsequent 5D / 20D return label.

---

## Institutional Attention Features

Features include:

- `research_events_30d`
- `research_events_90d`
- `institution_count_30d`
- `institution_count_90d`
- `max_institutions_30d`
- `days_since_last_research`
- `attention_acceleration`
- `attention_intensity_30d`
- `research_recency_score`

Highly skewed participation counts are log-transformed where appropriate.

### Incremental Attention

Institutional attention can naturally be related to large and liquid stocks.

To test whether the alternative data contains distinct information:

```text
Attention
= alpha
+ beta1 × Size
+ beta2 × Size-Neutral Liquidity
+ residual
```

The residual is treated as **incremental institutional attention**.

Exposure diagnostics showed:

```text
Attention vs Size                    = +0.152
Attention vs Size-Neutral Liquidity  = -0.041

Incremental Attention vs Size        = +0.007
Incremental Attention vs Liquidity   = -0.020
```

Raw and residualized institutional attention remained highly correlated:

```text
Spearman Corr = +0.958
```

This suggests that most variation in institutional attention was not simply a repackaging of Size or Liquidity.

---

## Alternative-Data Results

Because only **7 evaluable month-end observations** are available, these results are treated as **exploratory / hypothesis-generating**.

### 5-Day Horizon

| Feature | Mean Rank IC | Positive IC Ratio | Mean Top-Bottom Spread |
|---|---:|---:|---:|
| 30D Attention | +0.0373 | 57.1% | +0.66% |
| Incremental Attention | +0.0217 | 57.1% | +0.10% |
| 90D Attention | +0.0115 | 57.1% | -0.37% |

### 20-Day Horizon

| Feature | Mean Rank IC |
|---|---:|
| 30D Attention | -0.0310 |
| Incremental Attention | -0.0567 |
| 90D Attention | -0.0877 |

The short sample shows different behavior across horizons. The evidence is insufficient to claim stable predictive power.

---

## Project Structure

```text
a_share_quant_data_project/
│
├── config.py
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── outputs/
│
└── src/
    ├── 01_fetch_universe.py
    ├── 02_fetch_prices.py
    ├── 03_fetch_profiles.py
    ├── 03b_fill_missing_profiles.py
    ├── 04_fetch_fundamentals.py
    ├── 04b_audit_fundamentals.py
    ├── 04c_clean_fundamentals.py
    ├── 05_build_research_panel.py
    ├── 06_build_factor_features.py
    ├── 07_process_factors.py
    ├── 08_evaluate_factors.py
    ├── 09_fetch_valuation_data.py
    ├── 09b_build_size_value_factors.py
    ├── 10_evaluate_extended_factors.py
    ├── 11_fetch_institutional_research.py
    ├── 12_build_institutional_attention_features.py
    └── 13_evaluate_institutional_attention.py
```

---

## How to Run

### 1. Create an environment

Example with Conda:

```bash
conda create -n ashare-research python=3.11 -y
conda activate ashare-research
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python src/01_fetch_universe.py
python src/02_fetch_prices.py
python src/03_fetch_profiles.py
python src/03b_fill_missing_profiles.py
python src/04_fetch_fundamentals.py
python src/04b_audit_fundamentals.py
python src/04c_clean_fundamentals.py
python src/05_build_research_panel.py
python src/06_build_factor_features.py
python src/07_process_factors.py
python src/08_evaluate_factors.py
python src/09_fetch_valuation_data.py
python src/09b_build_size_value_factors.py
python src/10_evaluate_extended_factors.py
python src/11_fetch_institutional_research.py
python src/12_build_institutional_attention_features.py
python src/13_evaluate_institutional_attention.py
```

For long API downloads on macOS, `caffeinate` can be used to prevent idle sleep:

```bash
caffeinate -i python src/04_fetch_fundamentals.py
```

---

## Key Outputs

Examples under `outputs/`:

```text
fundamental_missing_summary.csv
fundamental_missing_periods.csv
fundamental_distribution_summary.csv
fundamental_extreme_rows.csv

factor_evaluation_summary.csv
factor_yearly_stability.csv
factor_correlation_matrix.csv

extended_factor_evaluation_summary.csv
liquidity_size_daily_correlation.csv
extended_factor_correlation.csv

institutional_attention_evaluation.csv
institutional_attention_factor_correlation.csv
```

---

## Limitations

This project intentionally documents its main limitations.

1. **Survivorship bias**  
   Current CSI 300 constituents are used retrospectively instead of historical point-in-time membership.

2. **Limited universe**  
   Only 80 stocks are included.

3. **Public data sources**  
   Public APIs are used rather than institutional Level-2 or licensed vendor feeds.

4. **Simplified industry neutralization**  
   Within-industry demeaning is used instead of a full Barra-style regression model.

5. **Size proxy**  
   Size is reconstructed using an implied free-float market-cap proxy rather than an official historical total-market-cap series.

6. **Alternative-data history**  
   Institutional-research history is limited to October 2025 onward, leaving only 7 usable month-end observations after the 90-day warm-up.

7. **Backtest realism**  
   The project evaluates cross-sectional predictive relationships rather than a fully executable portfolio. Transaction costs, turnover constraints, slippage, portfolio construction, and execution timing are not modeled.

---

## What This Project Demonstrates

The project is designed to demonstrate the full reasoning chain behind reliable quantitative research data:

```text
Raw Data
→ Data Quality
→ Business Logic
→ Point-in-Time Alignment
→ Research-Ready Dataset
→ Feature Engineering
→ Factor Standardization
→ Exposure Decomposition
→ Factor Evaluation
→ Alternative-Data Exploration
→ Explicit Limitation Management
```

The main takeaway is not that every factor produces alpha.

It is that a useful research pipeline must distinguish between:

- real signals and data-quality artifacts,
- observable information and look-ahead contamination,
- independent characteristics and hidden factor exposures,
- interesting patterns and statistically underpowered results.

That distinction is central to robust quantitative research and research-data engineering.
