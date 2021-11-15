import argparse
import os
import pickle as pkl

import torch

from attribution.mask import Mask
from attribution.perturbation import FadeMovingAverage
from fit.TSX.utils import load_data
from models.models import StateClassifier
from utils.losses import log_loss_target


def run_experiment(cv: int = 0, area: float = 0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "./data/mimic"
    p_data, train_loader, valid_loader, test_loader = load_data(batch_size=100, path=data_path, task="mortality", cv=cv)
    # Load the input time series:
    for X, Y in test_loader:
        X_test = X.to(device).type(torch.float32).transpose(1, 2)
        N_ex, T, N_features = X_test.shape
        Y_test = Y.type(torch.int64).to(device)

    # Save the test input and labels
    if not os.path.exists(f"./experiments/results/mimic/true_labels_{cv}.pkl"):
        with open(f"./experiments/results/mimic/true_labels_{cv}.pkl", "wb") as f:
            pkl.dump(Y_test, f)
    if not os.path.exists(f"./experiments/results/mimic/inputs_{cv}.pkl"):
        with open(f"./experiments/results/mimic/inputs_{cv}.pkl", "wb") as f:
            pkl.dump(X_test, f)

    # Load the model:
    model = StateClassifier(
        feature_size=N_features, n_state=2, hidden_size=200, rnn="GRU", return_all=True, device=device
    )
    model.load_state_dict(torch.load(f"experiments/results/mimic/model_{cv}.pt"))

    # This parts allow to use backprop on a RNN in evaluation mode (otherwise Pytorch crashes):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        elif isinstance(module, torch.nn.LSTM):
            module.dropout = 0
        elif isinstance(module, torch.nn.GRU):
            module.dropout = 0

    # Compute the classes predicted by the model
    Y = model(X_test.transpose(1, 2))
    Y = Y[:, -1]
    Y = Y.reshape(-1, 2)
    Y = torch.softmax(Y, dim=-1)
    Y = torch.argmax(Y, dim=-1)

    # The black-box is defined with the loaded model
    def f(x):
        x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        out = model(x)
        out = out[:, -1]
        out = torch.nn.Softmax()(out)
        return out

    # Prepare the useful variables:
    pert = FadeMovingAverage(device)  # This is the perturbation operator
    mask_saliency = torch.zeros(size=X_test.shape, dtype=torch.float32, device=device)  # This is the mask saliency map

    for k, x_test in enumerate(torch.unbind(X_test)):
        print(f"Now working with sample {k + 1}/{N_ex}.")
        # Fit the mask:
        mask = Mask(pert, device, task="classification", verbose=False, deletion_mode=True)
        mask.fit(
            X=x_test,
            f=f,
            loss_function=log_loss_target,
            keep_ratio=area,
            target=Y[k],
            learning_rate=1.0,
            size_reg_factor_init=0.1,
            size_reg_factor_dilation=10000,
            initial_mask_coeff=0.5,
            n_epoch=1000,
            momentum=1.0,
            time_reg_factor=0,
        )
        mask_saliency[k, :, :] = mask.mask_tensor

    # Convert the mask saliency to a numpy array, save it
    mask_saliency_np = mask_saliency.clone().detach().cpu().numpy()
    save_path = f"experiments/results/mimic/dynamask_test_importance_scores_{cv}_{int(10*area)}.pkl"
    with open(save_path, "wb") as file:
        pkl.dump(mask_saliency_np, file)
        print("Saving importance scores in " + save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", type=int, help="cross validation number of the experiment", default=0)
    parser.add_argument("--area", type=float, help="mask area", default=0.1)
    args = parser.parse_args()
    run_experiment(args.cv, args.area)
