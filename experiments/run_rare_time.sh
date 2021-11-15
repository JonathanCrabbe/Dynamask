# Rare > Feature.
cd "$(dirname "$0")/.."

# 1. Run
python -m experiments.rare_time --cv=0
python -m experiments.rare_time --cv=1
python -m experiments.rare_time --cv=2
python -m experiments.rare_time --cv=3
python -m experiments.rare_time --cv=4

# 2. Get Results
python -m experiments.results.rare_time.get_results --CV 5 --explainers dynamask fo fp ig shap
