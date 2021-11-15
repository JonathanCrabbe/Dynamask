import argparse
import os
import pickle as pkl

import numpy as np
import torch
from statsmodels.tsa.arima_process import ArmaProcess

from attribution.mask_group import MaskGroup
from attribution.perturbation import GaussianBlur
from baselines.explainers import FO, FP, IG, SVS
from utils.losses import mse

explainers = ["dynamask", "fo", "fp", "ig", "shap"]


def run_experiment(
    cv: int = 0,
    N_ex: int = 10,
    T: int = 50,
    N_features: int = 50,
    N_select: int = 5,
    save_dir: str = "experiments/results/rare_time/",
):
    """Run experiment.

    Args:
        cv: Do the experiment with different cv to obtain error bars.
        N_ex: Number of time series to generate.
        T: Length of each time series.
        N_features: Number of features in each time series.
        N_select: Number of time steps that are truly salient.
        save_dir: Directory where the results should be saved.

    Return:
        None
    """
    # Create the saving directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize useful variables
    random_seed = cv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    pert = GaussianBlur(device=device)  # We use a Gaussian Blur perturbation operator

    # Generate the input data
    ar = np.array([2, 0.5, 0.2, 0.1])  # AR coefficients
    ma = np.array([2])  # MA coefficients
    data_arima = ArmaProcess(ar=ar, ma=ma).generate_sample(nsample=(N_ex, T, N_features), axis=1)
    X = torch.tensor(data_arima, device=device, dtype=torch.float32)

    # Initialize the saliency tensors
    true_saliency = torch.zeros(size=(N_ex, T, N_features), device=device, dtype=torch.int64)
    dynamask_saliency = torch.zeros(size=true_saliency.shape, device=device)
    fo_saliency = torch.zeros(size=true_saliency.shape, device=device)
    fp_saliency = torch.zeros(size=true_saliency.shape, device=device)
    ig_saliency = torch.zeros(size=true_saliency.shape, device=device)
    shap_saliency = torch.zeros(size=true_saliency.shape, device=device)

    for k in range(N_ex):  # We compute the attribution for each individual time series
        print(f"Now working on example {k + 1}/{N_ex}.")
        # The truly salient times are selected randomly
        t_rand = np.random.randint(low=0, high=T - N_select)
        true_saliency[k, t_rand : t_rand + N_select, int(N_features / 4) : int(3 * N_features / 4)] = 1
        x = X[k, :, :]

        # The white box only depends on the truly salient features
        def f(input):
            output = torch.zeros(input.shape, device=input.device)
            output[t_rand : t_rand + N_select, int(N_features / 4) : int(3 * N_features / 4)] = input[
                t_rand : t_rand + N_select, int(N_features / 4) : int(3 * N_features / 4)
            ]
            output = (output ** 2).sum(dim=-1)
            return output

        # Dynamask attribution
        mask_group = MaskGroup(perturbation=pert, device=device, random_seed=random_seed, verbose=False)
        mask_group.fit(
            f=f,
            X=x,
            area_list=np.arange(0.001, 0.051, 0.001),
            loss_function=mse,
            n_epoch=1000,
            size_reg_factor_dilation=1000,
            size_reg_factor_init=1,
            learning_rate=1,
        )
        mask = mask_group.get_best_mask()
        dynamask_attr = mask.mask_tensor.clone().detach()
        dynamask_saliency[k, :, :] = dynamask_attr

        # Feature Occlusion attribution
        fo = FO(f=f)
        fo_attr = fo.attribute(x)
        fo_saliency[k, :, :] = fo_attr

        # Feature Permutation attribution
        fp = FP(f=f)
        fp_attr = fp.attribute(x)
        fp_saliency[k, :, :] = fp_attr

        # Integrated Gradient attribution
        ig = IG(f=f)
        ig_attr = ig.attribute(x)
        ig_saliency[k, :, :] = ig_attr

        # Sampling Shapley Value attribution
        shap = SVS(f=f)
        shap_attr = shap.attribute(x)
        shap_saliency[k, :, :] = shap_attr

    # Save everything in the directory
    with open(os.path.join(save_dir, f"true_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(true_saliency, file)
    with open(os.path.join(save_dir, f"dynamask_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(dynamask_saliency, file)
    with open(os.path.join(save_dir, f"fo_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(fo_saliency, file)
    with open(os.path.join(save_dir, f"fp_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(fp_saliency, file)
    with open(os.path.join(save_dir, f"ig_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(ig_saliency, file)
    with open(os.path.join(save_dir, f"shap_saliency_{cv}.pkl"), "wb") as file:
        pkl.dump(shap_saliency, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", default=0, type=int)
    args = parser.parse_args()
    run_experiment(cv=args.cv)
