import argparse
import pickle as pkl

import numpy as np
import torch
from sklearn import metrics
from torch.nn import Softmax

from attribution.mask_group import MaskGroup
from attribution.perturbation import GaussianBlur
from models.models import StateClassifier
from utils.losses import log_loss


def run_experiment(cv: int = 0):
    print(f"Welcome in the state experiment with cv = {cv} \n" + 100 * "=")
    softmax = Softmax(dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the input time series:
    with open("./data/state/state_dataset_x_test.pkl", "rb") as file:
        X_test = pkl.load(file)
        X_test = torch.from_numpy(X_test).to(device).type(torch.float32).transpose(1, 2)
        N_ex, T, N_features = X_test.shape

    # Load the true states that define the truly salient features and define A as in Section 3.2:
    with open("./data/state/state_dataset_states_test.pkl", "rb") as file:
        true_states = pkl.load(file)
        true_states += 1  #
        true_saliency = np.zeros((N_ex, T, N_features))  # Entries set to one for salient indices and zero else
        for exp_id, time_slice in enumerate(true_states):
            for t_id, feature_id in enumerate(time_slice):
                true_saliency[exp_id, t_id, feature_id] = 1
        true_saliency = true_saliency.astype(int)
    with open(f"./experiments/results/state/true_test_importance_{cv}.pkl", "wb") as file:
        pkl.dump(true_saliency, file)

    # Load the model:
    model = StateClassifier(feature_size=3, n_state=2, hidden_size=200, rnn="GRU", return_all=True)
    model.load_state_dict(torch.load(f"./experiments/results/state/model_{cv}.pt"))

    # This parts allow to use backprop on a RNN in evaluation mode (otherwise Pytorch crashes):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        elif isinstance(module, torch.nn.LSTM):
            module.dropout = 0
        elif isinstance(module, torch.nn.GRU):
            module.dropout = 0
    model.to(device)

    # The black-box is defined with the model:
    def f(x):
        x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        out = model(x)
        out = out.reshape(T, -1)
        out = softmax(out)
        return out

    # Prepare the useful variables:
    pert = GaussianBlur(device, sigma_max=1.0)  # This is the perturbation operator
    area_list = np.arange(0.25, 0.35, 0.01)  # This is the list of masks area to consider
    mask_saliency = torch.zeros(
        size=(N_ex, T, N_features), dtype=torch.float32, device=device
    )  # This is Dynamask's approximation for true_saliency

    for k, x_test in enumerate(torch.unbind(X_test)):
        print(f"Now working with sample {k + 1}/{N_ex}.")
        # Fit the group of mask:
        mask_group = MaskGroup(pert, device, verbose=False, deletion_mode=False)
        mask_group.fit(
            X=x_test,
            f=f,
            loss_function=log_loss,
            area_list=area_list,
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=100,
            initial_mask_coeff=0.5,
            n_epoch=1000,
            momentum=1.0,
            time_reg_factor=1.0,
        )

        # Extract the extremal mask:
        thresh = log_loss(f(x_test), f(x_test))  # This is what we call epsilon in the paper
        mask = mask_group.get_extremal_mask(threshold=thresh)
        mask_saliency[k, :, :] = mask.mask_tensor

        # Compute the metrics:
        prec, rec, thres = metrics.precision_recall_curve(
            true_saliency[k, :, :].flatten().astype(int), mask.mask_tensor.clone().detach().cpu().numpy().flatten()
        )
        print(
            f"For this iteration: AUP={metrics.auc(thres, prec[:-1]):.3g} ; AUR={metrics.auc(thres, rec[:-1]):.3g} ; "
            f"AUROC={metrics.roc_auc_score(true_saliency[k, :, :].flatten().astype(int), mask.mask_tensor.clone().detach().cpu().numpy().flatten()):.3g} ; "
            f"AUPRC={metrics.average_precision_score(true_saliency[k, :, :].flatten().astype(int), mask.mask_tensor.clone().detach().cpu().numpy().flatten()):.3g}\n"
            + 100 * "="
        )

    # Save the mask saliency map and print the metrics:
    mask_saliency_np = mask_saliency.clone().detach().cpu().numpy()
    save_path = f"./experiments/results/state/dynamask_test_importance_scores_{cv}.pkl"
    with open(save_path, "wb") as file:
        print(f"Saving the saliency scores in {save_path}.\n" + 100 * "=")
        pkl.dump(mask_saliency_np, file)

    mask_label = mask_saliency.clone().detach().cpu().numpy().flatten()
    true_label = true_saliency.flatten().astype(int)
    mask_prec, mask_rec, mask_thres = metrics.precision_recall_curve(true_label, mask_label)

    print(f"Mask AUROC: {metrics.roc_auc_score(true_label, mask_label)}")
    print(f"Mask AUPRC: {metrics.auc(mask_rec, mask_prec)}")
    print(f"Mask AUP: {metrics.auc(mask_thres, mask_prec[:-1])}")
    print(f"Mask AUR: {metrics.auc(mask_thres, mask_rec[:-1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, help="cross validation number of the experiment", default=0)
    args = parser.parse_args()
    run_experiment(args.cv)
