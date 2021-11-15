import argparse
import os
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn import metrics

from utils.metrics import get_entropy_array, get_information_array


def process_results(CV, explainers):
    metrics_array = np.zeros((len(explainers), 6, CV))  # This array contains the metrics for each method and each run
    path = "experiments/results/state"
    for cv in range(CV):
        # Load true saliency
        with open(os.path.join(path, f"true_test_importance_{cv}.pkl"), "rb") as file:
            true_saliency = pkl.load(file)
            true_saliency = true_saliency.flatten().astype(int)
        # Load the saliency map of each method
        for k, explainer in enumerate(explainers):
            with open(os.path.join(path, f"{explainer}_test_importance_scores_{cv}.pkl"), "rb") as file:
                saliency = pkl.load(file)
                if saliency.shape[1] < saliency.shape[2]:
                    saliency = saliency.transpose((0, 2, 1))
                if explainer in ["deep_lift", "integrated_gradient", "gradient_shap", "lime", "retain", "fo", "afo"]:
                    saliency = np.abs(saliency)  # Only the absolute value defines the importance in these cases
                # Normalize the saliency map:
                saliency -= saliency.min(axis=(1, 2), keepdims=True)
                saliency /= saliency.max(axis=(1, 2), keepdims=True)
                saliency = saliency.flatten()
                sub_saliency = saliency[true_saliency != 0]  # This is the saliency scores for each truly salient input

            # Compute all the relevant metrics, store them in the metric array
            prec, rec, thres = metrics.precision_recall_curve(true_saliency, saliency)
            metrics_array[k, 0, cv] = metrics.auc(thres, prec[:-1])
            metrics_array[k, 1, cv] = metrics.auc(thres, rec[:-1])
            metrics_array[k, 2, cv] = get_information_array(sub_saliency)
            metrics_array[k, 3, cv] = get_entropy_array(sub_saliency)
            metrics_array[k, 4, cv] = metrics.roc_auc_score(true_saliency, saliency)
            metrics_array[k, 5, cv] = metrics.average_precision_score(true_saliency, saliency)

    # The results array contain the average and standard deviation for each metric
    results = np.zeros((len(explainers), 12))
    results[:, 0::2] = metrics_array.mean(axis=2)
    results[:, 1::2] = metrics_array.std(axis=2)
    results_df = pd.DataFrame(
        data=results,
        index=explainers,
        columns=[
            "AUP",
            "AUP std",
            "AUR",
            "AUR std",
            "Info",
            "Info std",
            "Entr",
            "Entr std",
            "AUROC",
            "AUROC std",
            "AUPR",
            "AUPR std",
        ],
    )

    print(results_df)
    results_df.to_csv(os.path.join(path, "state_benchmark.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CV", default=1, help="Number of runs for the experiment.", type=int)
    parser.add_argument("--explainers", nargs="+", help="The explainers to include", type=str)
    args = parser.parse_args()
    process_results(args.CV, args.explainers)
