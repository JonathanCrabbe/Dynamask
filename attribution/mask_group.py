import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim

from attribution.mask import Mask
from attribution.perturbation import Perturbation


class MaskGroup:
    """This class allows to fit several mask of different areas simultaneously.

    Attributes:
        perturbation (attribution.perturbation.Perturbation):
            An object of the Perturbation class that uses the mask to generate perturbations.
        device: The device used to work with the torch tensors.
        verbose (bool): True is some messages should be displayed during optimization.
        random_seed (int): Random seed for reproducibility.
        deletion_mode (bool): True if the mask should identify the most impactful deletions.
        eps (float): Small number used for numerical stability.
        masks_tensor (torch.tensor): The tensor containing the coefficient of each mask
            (its size is len(area_list) * T * N_features).
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
        random_seed: int = 987,
        deletion_mode: bool = False,
        verbose: bool = True,
    ):
        self.perturbation = perturbation
        self.device = device
        self.random_seed = random_seed
        self.verbose = verbose
        self.deletion_mode = deletion_mode
        self.mask_list = None
        self.area_list = None
        self.f = None
        self.X = None
        self.n_epoch = None
        self.T = None
        self.N_features = None
        self.Y_target = None
        self.masks_tensor = None
        self.hist = None

    def fit(
        self,
        X,
        f,
        area_list,
        loss_function,
        n_epoch: int = 1000,
        initial_mask_coeff: float = 0.5,
        size_reg_factor_init: float = 0.1,
        size_reg_factor_dilation: float = 100,
        learning_rate: float = 0.1,
        momentum: float = 0.9,
        time_reg_factor: float = 0,
    ):
        """This method fits a group of masks to the input X for the black-box function f.

        Args:
            X: Input matrix (as a T*N_features torch tensor).
            f: Black-box (as a map compatible with torch tensors).
            area_list: The list of areas (a) of the masks we want to fit.
            loss_function: The loss function to optimize.
            n_epoch: Number of steps for the optimization.
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
        # Ensure that the area list is sorted
        area_list.sort()
        self.area_list = area_list
        N_area = len(area_list)
        # Create a list of masks
        mask_list = []
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
        self.Y_target = f(X)
        # The initial mask tensor has all coefficients set to initial_mask_coeff
        self.masks_tensor = initial_mask_coeff * torch.ones(size=(N_area, self.T, self.N_features), device=self.device)
        # The target is the same for each mask so we simply repeat it along the first axis
        Y_target_group = self.Y_target.clone().detach().unsqueeze(0).repeat(N_area, 1, 1)
        # Create a copy of the extremal tensor that is going to be trained, the optimizer and the history
        masks_tensor_new = self.masks_tensor.clone().detach().requires_grad_(True)
        optimizer = optim.SGD([masks_tensor_new], lr=learning_rate, momentum=momentum)
        hist = torch.zeros(3, 0)
        # Initializing the reference vector used in the regulator
        reg_ref = torch.ones((N_area, self.T * self.N_features), dtype=torch.float32, device=self.device)
        for i, area in enumerate(self.area_list):
            reg_ref[i, : int((1 - area) * self.T * self.N_features)] = 0.0
        # Run the optimization
        for k in range(n_epoch):
            # Measure the loop starting time
            t_loop = time.time()
            # Generate perturbed input and outputs
            if self.deletion_mode:
                X_pert = self.perturbation.apply_extremal(X=X, extremal_tensor=1 - masks_tensor_new)
            else:
                X_pert = self.perturbation.apply_extremal(X=X, extremal_tensor=masks_tensor_new)
            Y_pert = torch.stack([f(x_pert) for x_pert in torch.unbind(X_pert, dim=0)], dim=0)

            # Evaluate the overall loss (error [L_e] + size regulation [L_a] + time variation regulation [L_c])
            error = loss_function(Y_pert, Y_target_group)
            masks_tensor_sorted = masks_tensor_new.reshape(N_area, self.T * self.N_features).sort(dim=1)[0]
            size_reg = ((reg_ref - masks_tensor_sorted) ** 2).mean()
            time_reg = (torch.abs(masks_tensor_new[:, 1 : self.T - 1, :] - masks_tensor_new[:, : self.T - 2, :])).mean()
            loss = error_factor * error + reg_factor * size_reg + time_reg_factor * time_reg
            # Apply the gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Ensures that the constraint is fulfilled
            masks_tensor_new.data = masks_tensor_new.data.clamp(0, 1)
            # Save the error and the regulator
            metrics = torch.tensor([error, size_reg, time_reg]).cpu().unsqueeze(1)
            hist = torch.cat((hist, metrics), dim=1)
            # Increase the regulator coefficient
            reg_factor *= reg_multiplicator
            # Measure the loop ending time
            t_loop = time.time() - t_loop
            if self.verbose:
                print(
                    f"Epoch {k + 1}/{n_epoch}: error = {error.data:.3g} ; "
                    f"size regulator = {size_reg.data:.3g} ; time regulator = {time_reg.data:.3g} ;"
                    f" time elapsed = {t_loop:.3g} s"
                )
        # Update the mask and history tensor, print the final message
        self.masks_tensor = masks_tensor_new.clone().detach().requires_grad_(False)
        self.hist = hist
        t_fit = time.time() - t_fit
        print(
            f"The optimization finished: error = {error.data:.3g} ; size regulator = {size_reg.data:.3g} ;"
            f" time regulator = {time_reg.data:.3g} ; time elapsed = {t_fit:.3g} s"
        )

        # Store the individual mask coefficients in distinct mask objects
        for index, mask_tensor in enumerate(self.masks_tensor.unbind(dim=0)):
            mask = Mask(
                perturbation=self.perturbation, device=self.device, verbose=False, deletion_mode=self.deletion_mode
            )
            mask.mask_tensor = mask_tensor
            mask.hist = self.hist
            mask.f = self.f
            mask.X = self.X
            mask.n_epoch = self.n_epoch
            mask.T, mask.N_features = self.T, self.N_features
            mask.Y_target = self.Y_target
            mask.loss_function = loss_function
            mask_list.append(mask)
        self.mask_list = mask_list

    def get_best_mask(self):
        """This method returns the mask with lowest error."""
        error_list = [mask.get_error() for mask in self.mask_list]
        best_index = error_list.index(min(error_list))
        print(
            f"The mask of area {self.area_list[best_index]:.2g} is"
            f" the best with error = {error_list[best_index]:.3g}."
        )
        return self.mask_list[best_index]

    def get_extremal_mask(self, threshold):
        """This method returns the extremal mask for the acceptable error threshold (called epsilon in the paper)."""
        error_list = [mask.get_error() for mask in self.mask_list]
        # If the minimal error is above the threshold, the best we can do is select the mask with lowest error
        if min(error_list) > threshold:
            return self.get_best_mask()
        else:
            for id_mask, error in enumerate(error_list):
                if error < threshold:
                    print(
                        f"The mask of area {self.area_list[id_mask]:.2g} is"
                        f" extremal with error = {error_list[id_mask]:.3g}."
                    )
                    return self.mask_list[id_mask]

    def plot_errors(self):
        """This method plots the error as a function of the mask size."""
        sns.set()
        error_list = [mask.get_error() for mask in self.mask_list]
        plt.plot(self.area_list, error_list)
        plt.title("Errors for the various masks")
        plt.xlabel("Mask area")
        plt.ylabel("Error")
        plt.show()
