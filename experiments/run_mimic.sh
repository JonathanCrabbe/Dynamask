# MIMIC.

# Set your local MIMIC database (PostgreSQL) user and password:
YOUR_USER=myuser
YOUR_USER=mypassword


# 0. Get and prepare data
cd "$(dirname "$0")/.."
python fit/data_generator/icu_mortality.py --sqluser $YOUR_USER --sqlpass $YOUR_PASSWORD
python fit/data_generator/data_preprocess.py


# 1. Run
# -- cv=0
python -m fit.evaluation.baselines --data mimic --explainer fit --cv 0 --train
python -m fit.evaluation.baselines --data mimic --explainer lime --cv 0
python -m fit.evaluation.baselines --data mimic --explainer retain --cv 0 --train
python -m fit.evaluation.baselines --data mimic --explainer integrated_gradient --cv 0
python -m fit.evaluation.baselines --data mimic --explainer deep_lift --cv 0
python -m fit.evaluation.baselines --data mimic --explainer fo --cv 0
python -m fit.evaluation.baselines --data mimic --explainer afo --cv 0
python -m fit.evaluation.baselines --data mimic --explainer gradient_shap --cv 0
python -m experiments.mimic --cv 0 --area 0.1
python -m experiments.mimic --cv 0 --area 0.2
python -m experiments.mimic --cv 0 --area 0.3
python -m experiments.mimic --cv 0 --area 0.4
python -m experiments.mimic --cv 0 --area 0.5
python -m experiments.mimic --cv 0 --area 0.6
# -- cv=1
python -m fit.evaluation.baselines --data mimic --explainer fit --cv 1 --train
python -m fit.evaluation.baselines --data mimic --explainer lime --cv 1
python -m fit.evaluation.baselines --data mimic --explainer retain --cv 1 --train
python -m fit.evaluation.baselines --data mimic --explainer integrated_gradient --cv 1
python -m fit.evaluation.baselines --data mimic --explainer deep_lift --cv 1
python -m fit.evaluation.baselines --data mimic --explainer fo --cv 1
python -m fit.evaluation.baselines --data mimic --explainer afo --cv 1
python -m fit.evaluation.baselines --data mimic --explainer gradient_shap --cv 1
python -m experiments.mimic --cv 1 --area 0.1
python -m experiments.mimic --cv 1 --area 0.2
python -m experiments.mimic --cv 1 --area 0.3
python -m experiments.mimic --cv 1 --area 0.4
python -m experiments.mimic --cv 1 --area 0.5
python -m experiments.mimic --cv 1 --area 0.6
# -- cv=2
python -m fit.evaluation.baselines --data mimic --explainer fit --cv 2 --train
python -m fit.evaluation.baselines --data mimic --explainer lime --cv 2
python -m fit.evaluation.baselines --data mimic --explainer retain --cv 2 --train
python -m fit.evaluation.baselines --data mimic --explainer integrated_gradient --cv 2
python -m fit.evaluation.baselines --data mimic --explainer deep_lift --cv 2
python -m fit.evaluation.baselines --data mimic --explainer fo --cv 2
python -m fit.evaluation.baselines --data mimic --explainer afo --cv 2
python -m fit.evaluation.baselines --data mimic --explainer gradient_shap --cv 2
python -m experiments.mimic --cv 2 --area 0.1
python -m experiments.mimic --cv 2 --area 0.2
python -m experiments.mimic --cv 2 --area 0.3
python -m experiments.mimic --cv 2 --area 0.4
python -m experiments.mimic --cv 2 --area 0.5
python -m experiments.mimic --cv 2 --area 0.6


# 2. Get Results
python -m experiments.results.mimic.plot_benchmarks --CV 3 --explainers dynamask fo afo deep_lift fit gradient_shap integrated_gradient lime retain --areas 0.1 0.2 0.3 0.4 0.5 0.6
