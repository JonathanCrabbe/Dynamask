import argparse
import pickle as pkl

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve

from utils.metrics import get_entropy_array, get_information_array


def process_results(CV, explainer_list):
    metrics = np.zeros((4, len(explainer_list), CV))
    results_df = pd.DataFrame(columns=["AUP", "AUP std", "AUR", "AUR std", "Info", "Info std", "Entr", "Entr std"])
    for cv in range(CV):
        with open(f"experiments/results/rare_time/true_saliency_{cv}.pkl", "rb") as f:
            true_saliency = pkl.load(f).cpu().numpy()
        for e, explainer in enumerate(explainer_list):
            with open(f"experiments/results/rare_time/{explainer}_saliency_{cv}.pkl", "rb") as f:
                pred_saliency = pkl.load(f).cpu().numpy()
            prec, rec, thres = precision_recall_curve(true_saliency.flatten(), pred_saliency.flatten())
            metrics[0, e, cv] = auc(thres, prec[1:])
            metrics[1, e, cv] = auc(thres, rec[1:])
            # Normalize the saliency map:
            pred_saliency -= pred_saliency.min(axis=(1, 2), keepdims=True)
            pred_saliency /= pred_saliency.max(axis=(1, 2), keepdims=True)
            sub_saliency = pred_saliency[true_saliency != 0]  # This is the saliency scores for each truly salient input
            metrics[2, e, cv] = get_information_array(sub_saliency, eps=1.0e-5)
            metrics[3, e, cv] = get_entropy_array(sub_saliency, eps=1.0e-5)

    for e, explainer in enumerate(explainer_list):
        aup_avg, aup_std = np.mean(metrics[0, e, :]), np.std(metrics[0, e, :])
        aur_avg, aur_std = np.mean(metrics[1, e, :]), np.std(metrics[1, e, :])
        im_avg, im_std = np.mean(metrics[2, e, :]), np.std(metrics[2, e, :])
        sm_avg, sm_std = np.mean(metrics[3, e, :]), np.std(metrics[3, e, :])
        results_df.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std, im_avg, im_std, sm_avg, sm_std]

    print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CV", default=1, help="Number of runs for the experiment.", type=int)
    parser.add_argument("--explainers", nargs="+", help="The explainers to include", type=str)
    args = parser.parse_args()
    process_results(args.CV, args.explainers)
