# Rare > Feature.
cd "$(dirname "$0")/.."

# 1. Run
python -m experiments.rare_feature --cv=0
python -m experiments.rare_feature --cv=1
python -m experiments.rare_feature --cv=2
python -m experiments.rare_feature --cv=3
python -m experiments.rare_feature --cv=4

# 2. Get Results
python -m experiments.results.rare_feature.get_results --CV 5 --explainers dynamask fo fp ig shap
