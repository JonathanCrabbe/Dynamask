import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from torch.nn import Softmax

from attribution.perturbation import Perturbation
from utils.metrics import get_entropy, get_information


class Mask:
    """This class allows to fit and interact with dynamic masks.

    Attributes:
        perturbation (attribution.perturbation.Perturbation):
            An object of the Perturbation class that uses the mask to generate perturbations.
        device: The device used to work with the torch tensors.
        verbose (bool): True is some messages should be displayed during optimization.
        random_seed (int): Random seed for reproducibility.
        deletion_mode (bool): True if the mask should identify the most impactful deletions.
        eps (float): Small number used for numerical stability.
        mask_tensor (torch.tensor): The tensor containing the mask coefficients.
        T (int): Number of time steps.
        N_features (int): Number of features.
        Y_target (torch.tensor): Black-box prediction.
        hist (torch.tensor): History tensor containing the metrics at different epochs.
        task (str): "classification" or "regression".
    """

    def __init__(
        self,
        perturbation: Perturbation,
        device,
        task: str = "regression",
        verbose: bool = False,
        random_seed: int = 42,
        deletion_mode: bool = False,
        eps: float = 1.0e-7,
    ):

        self.verbose = verbose
        self.device = device
        self.random_seed = random_seed
        self.deletion_mode = deletion_mode
        self.perturbation = perturbation
        self.eps = eps
        self.task = task
        self.X = None
        self.mask_tensor = None
        self.T = None
        self.N_features = None
        self.Y_target = None
        self.f = None
        self.n_epoch = None
        self.hist = None
        self.loss_function = None

    # Mask Optimization

    def fit(
        self,
        X,
        f,
        loss_function,
        target=None,
        n_epoch: int = 500,
        keep_ratio: float = 0.5,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.5,
        size_reg_factor_dilation: float = 100,
        time_reg_factor: float = 0,
        learning_rate: float = 1.0e-1,
        momentum: float = 0.9,
    ):
        """This method fits a mask to the input X for the black-box function f.

        Args:
            X: Input matrix (as a T*N_features torch tensor).
            f: Black-box (as a map compatible with torch tensors).
            loss_function: The loss function to optimize.
            target: If the output to approximate is different from f(X), it can be specified optionally.
            n_epoch: Number of steps for the optimization.
            keep_ratio: Fraction of elements in X that should be kept by the mask (called a in the paper).
            initial_mask_coeff: Initial value for the mask coefficient (called lambda_0 in the paper).
            size_reg_factor_init: Initial coefficient for the regulator part of the total loss.
            size_reg_factor_dilation: Ratio between the final and the initial size regulation factor
                (called delta in the paper).
            time_reg_factor: Regulation factor for the variation in time (called lambda_a in the paper).
            learning_rate: Learning rate for the torch SGD optimizer.
            momentum: Momentum for the SGD optimizer.

        Returns:
            None
        """
        # Initialize the random seed and the attributes
        t_fit = time.time()
        torch.manual_seed(self.random_seed)
        reg_factor = size_reg_factor_init
        error_factor = 1 - 2 * self.deletion_mode  # In deletion mode, the error has to be maximized
        reg_multiplicator = np.exp(np.log(size_reg_factor_dilation) / n_epoch)
        self.f = f
        self.X = X
        self.n_epoch = n_epoch
        self.T, self.N_features = X.shape
        self.loss_function = loss_function
        if target is None:
            self.Y_target = f(X)
        else:
            self.Y_target = target

        # The initial mask is defined with the initial mask coefficient
        self.mask_tensor = initial_mask_coeff * torch.ones(size=X.shape, device=self.device)
        # Create a copy of the mask that is going to be trained, the optimizer and the history
        mask_tensor_new = self.mask_tensor.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([mask_tensor_new], lr=learning_rate, momentum=momentum)
        hist = torch.zeros(3, 0)
        # Initializing the reference vector used in the size regulator (called r_a in the paper)
        reg_ref = torch.zeros(int((1 - keep_ratio) * self.T * self.N_features))
        reg_ref = torch.cat((reg_ref, torch.ones(self.T * self.N_features - reg_ref.shape[0]))).to(self.device)

        # Run the optimization
        for k in range(n_epoch):
            # Measure the loop starting time
            t_loop = time.time()
            # Generate perturbed input and outputs
            if self.deletion_mode:
                X_pert = self.perturbation.apply(X=X, mask_tensor=1 - mask_tensor_new)
            else:
                X_pert = self.perturbation.apply(X=X, mask_tensor=mask_tensor_new)
            Y_pert = f(X_pert)
            # Evaluate the overall loss (error [L_e] + size regulation [L_a] + time variation regulation [L_c])
            error = loss_function(Y_pert, self.Y_target)
            mask_tensor_sorted = mask_tensor_new.reshape(self.T * self.N_features).sort()[0]
            size_reg = ((reg_ref - mask_tensor_sorted) ** 2).mean()
            time_reg = (torch.abs(mask_tensor_new[1 : self.T - 1, :] - mask_tensor_new[: self.T - 2, :])).mean()
            loss = error_factor * error + reg_factor * size_reg + time_reg_factor * time_reg
            # Apply the gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Ensures that the constraint is fulfilled
            mask_tensor_new.data = mask_tensor_new.data.clamp(0, 1)
            # Save the error and the regulator
            metrics = torch.tensor([error.detach().cpu(), size_reg.detach().cpu(), time_reg.detach().cpu()]).unsqueeze(
                1
            )
            hist = torch.cat((hist, metrics), dim=1)
            # Increase the regulator coefficient
            reg_factor *= reg_multiplicator
            # Measure the loop ending time
            t_loop = time.time() - t_loop
            if self.verbose:
                print(
                    f"Epoch {k + 1}/{n_epoch}: error = {error.data:.3g} ; size regulator = {size_reg.data:.3g} ;"
                    f" time regulator = {time_reg.data:.3g} ; time elapsed = {t_loop:.3g} s"
                )
        # Update the mask and history tensor, print the final message
        self.mask_tensor = mask_tensor_new
        self.hist = hist
        t_fit = time.time() - t_fit
        print(
            100 * "="
            + "\n"
            + f"The optimization finished: error = {error.data:.3g} ; size regulator = {size_reg.data:.3g} ;"
            f" time regulator = {time_reg.data:.3g} ; time elapsed = {t_fit:.3g} s" + "\n" + 100 * "=" + "\n"
        )

    # Mask Manipulation

    def get_smooth_mask(self, sigma=1):
        """This method smooths the mask tensor by applying a temporal Gaussian filter for each feature.

        Args:
            sigma: Width of the Gaussian filter.

        Returns:
            torch.Tensor: The smoothed mask.
        """
        # Define the Gaussian smoothing kernel
        T_axis = torch.arange(1, self.T + 1, dtype=int, device=self.device)
        T1_tensor = T_axis.unsqueeze(1).unsqueeze(2)
        T2_tensor = T_axis.unsqueeze(0).unsqueeze(2)
        kernel_tensor = torch.exp(-1.0 * (T1_tensor - T2_tensor) ** 2 / (2.0 * sigma ** 2))
        kernel_tensor = torch.divide(kernel_tensor, torch.sum(kernel_tensor, 0))
        kernel_tensor = kernel_tensor.repeat(1, 1, self.N_features)
        # Smooth the mask tensor by applying the kernel
        mask_tensor_smooth = torch.einsum("sti,si->ti", kernel_tensor, self.mask_tensor)
        return mask_tensor_smooth

    def extract_submask(self, mask_tensor, ids_time, ids_feature):
        """This method extracts a submask specified with specified indices.

        Args:
            mask_tensor: The tensor from which data should be extracted.
            ids_time: List of the times that should be extracted.
            ids_feature: List of the features that should be extracted.

        Returns:
            torch.Tensor: Submask extracted based on the indices.
        """
        # If no identifiers have been specified, we use the whole data
        if ids_time is None:
            ids_time = [k for k in range(self.T)]
        if ids_feature is None:
            ids_feature = [k for k in range(self.N_features)]
        # Extract the relevant data in the mask
        submask_tensor = mask_tensor.clone().detach().requires_grad_(False).cpu()
        submask_tensor = submask_tensor[ids_time, :]
        submask_tensor = submask_tensor[:, ids_feature]
        return submask_tensor

    # Mask plots

    def plot_mask(self, ids_time=None, ids_feature=None, smooth: bool = False, sigma: float = 1.0):
        """This method plots (part of) the mask.

        Args:
            ids_time: List of the times that should appear on the plot.
            ids_feature: List of the features that should appear on the plot.
            smooth: True if the mask should be smoothed before plotting.
            sigma: Width of the smoothing Gaussian kernel.

        Returns:
            None
        """
        sns.set()
        # Smooth the mask if required
        if smooth:
            mask_tensor = self.get_smooth_mask(sigma)
        else:
            mask_tensor = self.mask_tensor
        # Extract submask from ids
        submask_tensor_np = self.extract_submask(mask_tensor, ids_time, ids_feature).numpy()
        df = pd.DataFrame(data=np.transpose(submask_tensor_np), index=ids_feature, columns=ids_time)
        # Generate heatmap plot
        color_map = sns.diverging_palette(10, 133, as_cmap=True)
        heat_map = sns.heatmap(data=df, cmap=color_map, cbar_kws={"label": "Mask"}, vmin=0, vmax=1)
        plt.xlabel("Time")
        plt.ylabel("Feature Number")
        plt.title("Mask coefficients over time")
        plt.show()

    def plot_hist(self):
        """This method plots the metrics for different epochs of optimization."""
        if self.hist is None:
            raise RuntimeError("The mask should be optimized before plotting the metrics.")
        sns.set()
        # Extract the error and regulator history from the history tensor
        error, size_reg, time_reg = self.hist[:].clone().detach().cpu().numpy()
        epoch_axis = np.arange(1, len(error) + 1)
        # Generate the subplots
        fig, axs = plt.subplots(3)
        axs[0].plot(epoch_axis, error)
        axs[0].set(xlabel="Epoch", ylabel="Error")
        axs[1].plot(epoch_axis, size_reg)
        axs[1].set(xlabel="Epoch", ylabel="Size Regulator")
        axs[2].plot(epoch_axis, time_reg)
        axs[2].set(xlabel="Epoch", ylabel="Time Regulator")
        plt.show()

    # Mask metrics

    def get_information(self, ids_time=None, ids_feature=None, normalize: bool = False):
        """This methods returns the mask information contained in the identifiers.

        Args:
            normalize: Whether to normalize.
            ids_time: List of the times that should contribute.
            ids_feature: List of the features that should contribute.

        Returns:
            Information content as a torch scalar.
        """
        return get_information(
            self.mask_tensor, ids_time=ids_time, ids_feature=ids_feature, normalize=normalize, eps=self.eps
        )

    def get_entropy(self, ids_time=None, ids_feature=None, normalize: bool = False):
        """This methods returns the mask entropy contained in the identifiers.

        Args:
            normalize: Whether to normalize.
            ids_time: List of the times that should contribute.
            ids_feature: List of the features that should contribute.

        Returns:
            Entropy as a torch scalar.
        """
        return get_entropy(
            self.mask_tensor, ids_time=ids_time, ids_feature=ids_feature, normalize=normalize, eps=self.eps
        )

    def get_error(self):
        """This methods returns the error between the unperturbed and perturbed input [L_e].

        Returns:
            Error as a torch scalar.
        """
        if self.deletion_mode:
            X_pert = self.perturbation.apply(X=self.X, mask_tensor=1 - self.mask_tensor)
        else:
            X_pert = self.perturbation.apply(X=self.X, mask_tensor=self.mask_tensor)
        Y_pert = self.f(X_pert)
        if self.task == "classification":
            Y_pert = torch.log(Softmax(dim=1)(Y_pert))
        return self.loss_function(Y_pert, self.Y_target)
