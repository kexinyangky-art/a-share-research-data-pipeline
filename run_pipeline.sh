#!/bin/bash
set -e

python src/01_fetch_universe.py
python src/02_fetch_prices.py
python src/03_fetch_profiles.py
python src/04_fetch_fundamentals.py
python src/05_build_features.py
python src/06_quality_checks.py
python src/07_factor_analysis.py
