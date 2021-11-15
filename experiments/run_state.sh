# State.
cd "$(dirname "$0")/.."

# 0. Generate.
python -m fit.data_generator.state_data --signal_len 200 --signal_num 1000

# 1. Run
# -- cv=0
python -m fit.evaluation.baselines --explainer fit --cv 0 --train
python -m fit.evaluation.baselines --explainer lime --cv 0 --train
python -m fit.evaluation.baselines --explainer retain --cv 0 --train
python -m fit.evaluation.baselines --explainer integrated_gradient --cv 0 --train
python -m fit.evaluation.baselines --explainer deep_lift --cv 0 --train
python -m fit.evaluation.baselines --explainer fo --cv 0 --train
python -m fit.evaluation.baselines --explainer afo --cv 0 --train
python -m fit.evaluation.baselines --explainer gradient_shap --cv 0 --train
python -m experiments.state --cv 0
# -- cv=1
python -m fit.evaluation.baselines --explainer fit --cv 1 --train
python -m fit.evaluation.baselines --explainer lime --cv 1
python -m fit.evaluation.baselines --explainer retain --cv 1 --train
python -m fit.evaluation.baselines --explainer integrated_gradient --cv 1
python -m fit.evaluation.baselines --explainer deep_lift --cv 1
python -m fit.evaluation.baselines --explainer fo --cv 1
python -m fit.evaluation.baselines --explainer afo --cv 1
python -m fit.evaluation.baselines --explainer gradient_shap --cv 1
python -m experiments.state --cv 1
# -- cv=2
python -m fit.evaluation.baselines --explainer fit --cv 2 --train
python -m fit.evaluation.baselines --explainer lime --cv 2
python -m fit.evaluation.baselines --explainer retain --cv 2 --train
python -m fit.evaluation.baselines --explainer integrated_gradient --cv 2
python -m fit.evaluation.baselines --explainer deep_lift --cv 2
python -m fit.evaluation.baselines --explainer fo --cv 2
python -m fit.evaluation.baselines --explainer afo --cv 2
python -m fit.evaluation.baselines --explainer gradient_shap --cv 2
python -m experiments.state --cv 2
# -- cv=3
python -m fit.evaluation.baselines --explainer fit --cv 3 --train
python -m fit.evaluation.baselines --explainer lime --cv 3
python -m fit.evaluation.baselines --explainer retain --cv 3 --train
python -m fit.evaluation.baselines --explainer integrated_gradient --cv 3
python -m fit.evaluation.baselines --explainer deep_lift --cv 3
python -m fit.evaluation.baselines --explainer fo --cv 3
python -m fit.evaluation.baselines --explainer afo --cv 3
python -m fit.evaluation.baselines --explainer gradient_shap --cv 3
python -m experiments.state --cv 3
# -- cv=4
python -m fit.evaluation.baselines --explainer fit --cv 4 --train
python -m fit.evaluation.baselines --explainer lime --cv 4
python -m fit.evaluation.baselines --explainer retain --cv 4 --train
python -m fit.evaluation.baselines --explainer integrated_gradient --cv 4
python -m fit.evaluation.baselines --explainer deep_lift --cv 4
python -m fit.evaluation.baselines --explainer fo --cv 4
python -m fit.evaluation.baselines --explainer afo --cv 4
python -m fit.evaluation.baselines --explainer gradient_shap --cv 4
python -m experiments.state --cv 4

# 2. Get Results
python -m experiments.results.state.get_results --CV 5 --explainers dynamask fo afo deep_lift fit gradient_shap integrated_gradient lime retain
