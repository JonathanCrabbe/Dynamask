import pickle as pkl
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import auc, precision_recall_curve


def process_results(CV, explainer_list):
    metrics = np.zeros((2, len(explainer_list), CV))
    results_df = pd.DataFrame(columns=['AUP', 'AUP std', 'AUR', 'AUR std'])
    for cv in range(CV):
        with open(f'experiments/results/rare_time/true_saliency_{cv}.pkl', 'rb') as f:
            true_saliency = pkl.load(f).cpu().numpy()
        for e, explainer in enumerate(explainer_list):
            with open(f'experiments/results/rare_time/{explainer}_saliency_{cv}.pkl', 'rb') as f:
                pred_saliency = pkl.load(f).cpu().numpy()
            prec, rec, thres = precision_recall_curve(true_saliency.flatten(), pred_saliency.flatten())
            metrics[0, e, cv] = auc(thres, prec[1:])
            metrics[1, e, cv] = auc(thres, rec[1:])

    for e, explainer in enumerate(explainer_list):
        aup_avg, aup_std = np.mean(metrics[0, e, :]), np.std(metrics[0, e, :])
        aur_avg, aur_std = np.mean(metrics[1, e, :]), np.std(metrics[1, e, :])
        results_df.loc[explainer] = [aup_avg, aup_std, aur_avg, aur_std]

    print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--CV', default=1, help='Number of runs for the experiment.', type=int)
    parser.add_argument('--explainers', nargs='+', help='The explainers to include', type=str)
    args = parser.parse_args()
    process_results(args.CV, args.explainers)



