import argparse
import os
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn import metrics

from models.models import StateClassifier


def process_results(CV, explainers, areas):

    # Set parameters, load the relevant data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    areas = [float(area) for area in areas]
    N_metric = 2
    sns.set(font_scale=1.0)
    sns.set_style("white")
    plt.rcParams["pdf.fonttype"] = 42
    name_dict = {
        "fit": "FIT",
        "deep_lift": "DL",
        "afo": "AFO",
        "fo": "FO",
        "retain": "RT",
        "integrated_gradient": "IG",
        "gradient_shap": "GS",
        "lime": "LIME",
        "dynamask": "MASK",
    }
    metrics_array = np.zeros((len(explainers), len(areas), N_metric, CV))
    path = "./experiments/results/mimic"
    for cv in range(CV):
        with open(os.path.join(path, f"true_labels_{cv}.pkl"), "rb") as file:
            Y_true = pkl.load(file).cpu().numpy()
        with open(os.path.join(path, f"inputs_{cv}.pkl"), "rb") as file:
            X = pkl.load(file).to(device)
            X_avg = X.mean(dim=1, keepdim=True)
            N_exp, T, N_features = X.shape

        # Load the model:
        model = StateClassifier(
            feature_size=N_features, n_state=2, hidden_size=200, rnn="GRU", device=device, return_all=True
        )
        model.load_state_dict(torch.load(os.path.join(path, f"model_{cv}.pt")))

        # This parts allow to use backprop on a RNN in evaluation mode (otherwise Pytorch crashes):
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0
            elif isinstance(module, torch.nn.LSTM):
                module.dropout = 0
            elif isinstance(module, torch.nn.GRU):
                module.dropout = 0

        # For each mask area, we compute the CE and the ACC for each attribution method:
        for i, fraction in enumerate(areas):
            N_drop = int(fraction * N_exp * N_features * T)  # The number of inputs to perturb
            Y = model(X.transpose(1, 2))
            Y = Y[:, -1]
            Y = Y.reshape(-1, 2)
            Y = torch.softmax(Y, dim=-1)
            Y = torch.argmax(Y, dim=-1).detach().cpu().numpy()  # This is the predicted class for the unperturbed input

            # For each attribution method, use the saliency map to construct a perturbed input:
            for k, explainer in enumerate(explainers):
                if explainer == "dynamask":
                    with open(
                        os.path.join(path, f"dynamask_test_importance_scores_{cv}_{int(fraction*10)}.pkl"), "rb"
                    ) as file:
                        saliency = pkl.load(file)
                        if saliency.shape[1] < saliency.shape[2]:
                            saliency = saliency.transpose((0, 2, 1))
                        saliency_tensor = torch.from_numpy(saliency)
                        # Extract the N_drop inputs with maximal saliency that are going to be perturbed:
                        _, id_drop = torch.topk(saliency_tensor.flatten(), k=N_drop)
                        id_drop = np.array(np.unravel_index(id_drop.numpy(), saliency_tensor.shape))
                        mask_tensor = torch.zeros(size=X.shape, device=device, dtype=torch.float32)
                        mask_tensor[id_drop] = 1.0
                        # Perturb the most relevant inputs and compute the associated output:
                        X_pert = (1 - mask_tensor) * X + mask_tensor * X_avg
                        Y_pert = model(X_pert.transpose(1, 2))
                        Y_pert = Y_pert[:, -1]
                        Y_pert = Y_pert.reshape(-1, 2)
                        Y_pert = torch.softmax(Y_pert, dim=-1)
                        proba_pert = Y_pert.detach().cpu().numpy()
                        Y_pert = torch.argmax(Y_pert, dim=-1).detach().cpu().numpy()
                        metrics_array[k, i, 0, cv] = metrics.log_loss(Y, proba_pert)  # This is CE
                        metrics_array[k, i, 1, cv] = metrics.accuracy_score(Y, Y_pert)  # This is ACC

                else:
                    with open(os.path.join(path, f"{explainer}_test_importance_scores_{cv}.pkl"), "rb") as file:
                        saliency = pkl.load(file)
                        if saliency.shape[1] < saliency.shape[2]:
                            saliency = saliency.transpose((0, 2, 1))  # Reshape fit tensors to match masks shape
                        if explainer in [
                            "deep_lift",
                            "integrated_gradient",
                            "gradient_shap",
                            "lime",
                            "retain",
                            "fo",
                            "afo",
                        ]:
                            saliency = np.abs(saliency)  # Only the absolute value defines the importance in these cases
                        saliency_tensor = torch.from_numpy(saliency)
                        # Extract the N_drop inputs with maximal saliency that are going to be perturbed:
                        _, id_drop = torch.topk(saliency_tensor.flatten(), k=N_drop)
                        id_drop = np.array(np.unravel_index(id_drop.numpy(), saliency_tensor.shape))
                        mask_tensor = torch.zeros(size=X.shape, device=device, dtype=torch.float32)
                        mask_tensor[id_drop] = 1.0
                        # Perturb the most relevant inputs and compute the associated output:
                        X_pert = (1 - mask_tensor) * X + mask_tensor * X_avg
                        Y_pert = model(X_pert.transpose(1, 2))
                        Y_pert = Y_pert[:, -1]
                        Y_pert = Y_pert.reshape(-1, 2)
                        Y_pert = torch.softmax(Y_pert, dim=-1)
                        proba_pert = Y_pert.detach().cpu().numpy()
                        Y_pert = torch.argmax(Y_pert, dim=-1).detach().cpu().numpy()
                        metrics_array[k, i, 0, cv] = metrics.log_loss(Y, proba_pert)
                        metrics_array[k, i, 1, cv] = metrics.accuracy_score(Y, Y_pert)  # This is ACC

    # Plot the CE and the ACC for each attribution method and each mask area
    for k, name in enumerate(explainers):
        plt.figure(1)
        plt.plot(areas, metrics_array[k, :, 0, :].mean(axis=-1), label=name_dict[name])
        plt.fill_between(
            areas,
            metrics_array[k, :, 0, :].mean(axis=-1) - metrics_array[k, :, 0, :].std(axis=-1),
            metrics_array[k, :, 0, :].mean(axis=-1) + metrics_array[k, :, 0, :].std(axis=-1),
            alpha=0.2,
        )
        plt.figure(2)
        plt.plot(areas, metrics_array[k, :, 1, :].mean(axis=-1), label=name_dict[name])
        plt.fill_between(
            areas,
            metrics_array[k, :, 1, :].mean(axis=-1) - metrics_array[k, :, 1, :].std(axis=-1),
            metrics_array[k, :, 1, :].mean(axis=-1) + metrics_array[k, :, 1, :].std(axis=-1),
            alpha=0.2,
        )

    plt.figure(1)
    plt.xlabel("Fraction of the input perturbed")
    plt.ylabel("CE")
    plt.legend()
    plt.savefig(os.path.join(path, "ce.pdf"), bbox_inches="tight")

    plt.figure(2)
    plt.xlabel("Fraction of the input perturbed")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join(path, "acc.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--CV", default=1, help="Number of runs for the experiment.", type=int)
    parser.add_argument("--explainers", nargs="+", help="The explainers to include", type=str)
    parser.add_argument("--areas", nargs="+", help="The fractions of perturbed input", type=str)
    args = parser.parse_args()
    process_results(args.CV, args.explainers, args.areas)
