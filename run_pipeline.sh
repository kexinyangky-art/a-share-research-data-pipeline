#!/bin/bash

set -euo pipefail

echo "========================================"
echo "A-Share Research Data Pipeline"
echo "========================================"

echo "[1/16] Fetching research universe..."
python src/01_fetch_universe.py

echo "[2/16] Fetching daily market prices..."
python src/02_fetch_prices.py

echo "[3/16] Fetching company profiles..."
python src/03_fetch_profiles.py

echo "[4/16] Filling missing profile fields..."
python src/03b_fill_missing_profiles.py

echo "[5/16] Fetching quarterly fundamentals..."
python src/04_fetch_fundamentals.py

echo "[6/16] Auditing fundamental data..."
python src/04b_audit_fundamentals.py

echo "[7/16] Cleaning fundamental data..."
python src/04c_clean_fundamentals.py

echo "[8/16] Building point-in-time research panel..."
python src/05_build_research_panel.py

echo "[9/16] Building raw factor features..."
python src/06_build_factor_features.py

echo "[10/16] Processing factors..."
python src/07_process_factors.py

echo "[11/16] Evaluating core factors..."
python src/08_evaluate_factors.py

echo "[12/16] Fetching historical valuation data..."
python src/09_fetch_valuation_data.py

echo "[13/16] Building Size and Value factors..."
python src/09b_build_size_value_factors.py

echo "[14/16] Evaluating extended factors..."
python src/10_evaluate_extended_factors.py

echo "[15/16] Fetching institutional research data..."
python src/11_fetch_institutional_research.py

echo "[16/16] Building and evaluating alternative-data features..."
python src/12_build_institutional_attention_features.py
python src/13_evaluate_institutional_attention.py

echo "========================================"
echo "Pipeline completed successfully."
echo "========================================"
