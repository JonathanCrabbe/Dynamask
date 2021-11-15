from abc import ABC, abstractmethod

import torch


class Perturbation(ABC):
    """This class allows to create and apply perturbation on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor):
            The mask tensor than indicates the intensity of the perturbation to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
    """

    @abstractmethod
    def __init__(self, device, eps=1.0e-7):
        self.mask_tensor = None
        self.eps = eps
        self.device = device

    @abstractmethod
    def apply(self, X, mask_tensor):
        """This method applies the perturbation on the input based on the mask tensor.

        Args:
            X: Input tensor.
            mask_tensor: Tensor containing the mask coefficients.
        """
        if X is None or mask_tensor is None:
            raise NameError("The mask_tensor should be fitted before or while calling the perturb method.")

    @abstractmethod
    def apply_extremal(self, X, extremal_tensor: torch.Tensor):
        """This method applies the perturbation on the input based on the extremal tensor.

        The extremal tensor is just a set of mask, the perturbation is applied according to each mask.

        Args:
            X: Input tensor.
            extremal_tensor: (N_area, T, N_feature) tensor containing the different masks.
        """
        if X is None or extremal_tensor is None:
            raise NameError("The mask_tensor should be fitted before or while calling the perturb method.")


class FadeMovingAverage(Perturbation):
    """This class allows to create and apply 'fade to moving average' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
    """

    def __init__(self, device, eps=1.0e-7):
        super().__init__(eps=eps, device=device)

    def apply(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        T = X.shape[0]
        # Compute the moving average for each feature and concatenate it to create a tensor with X's shape
        moving_average = torch.mean(X, 0).reshape(1, -1).to(self.device)
        moving_average_tiled = moving_average.repeat(T, 1)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = mask_tensor * X + (1 - mask_tensor) * moving_average_tiled
        return X_pert

    def apply_extremal(self, X, extremal_tensor: torch.Tensor):
        super().apply_extremal(X, extremal_tensor)
        # Compute the moving average for each feature and concatenate it to create a tensor with X's shape
        moving_average = torch.mean(X, dim=0).reshape(1, 1, -1).to(self.device)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = extremal_tensor * X + (1 - extremal_tensor) * moving_average
        return X_pert


class GaussianBlur(Perturbation):
    """This class allows to create and apply 'Gaussian blur' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        sigma_max (float): Maximal width for the Gaussian blur.
    """

    def __init__(self, device, eps=1.0e-7, sigma_max=2):
        super().__init__(eps=eps, device=device)
        self.sigma_max = sigma_max

    def apply(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        T = X.shape[0]
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # Convert the mask into a tensor containing the width of each Gaussian perturbation
        sigma_tensor = self.sigma_max * ((1 + self.eps) - mask_tensor)
        sigma_tensor = sigma_tensor.unsqueeze(0)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.unsqueeze(1).unsqueeze(2)
        T2_tensor = T_axis.unsqueeze(0).unsqueeze(2)
        filter_coefs = torch.exp(torch.divide(-1.0 * (T1_tensor - T2_tensor) ** 2, 2.0 * (sigma_tensor ** 2)))
        filter_coefs = torch.divide(filter_coefs, torch.sum(filter_coefs, 0))
        # The perturbation is obtained by replacing each input by the linear combination weighted by Gaussian coefs
        X_pert = torch.einsum("sti,si->ti", filter_coefs, X)
        return X_pert

    def apply_extremal(self, X: torch.Tensor, extremal_tensor: torch.Tensor):
        N_area, T, N_features = extremal_tensor.shape
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # Convert the mask into a tensor containing the width of each Gaussian perturbation
        sigma_tensor = self.sigma_max * ((1 + self.eps) - extremal_tensor).reshape(N_area, 1, T, N_features)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.reshape(1, 1, T, 1)
        T2_tensor = T_axis.reshape(1, T, 1, 1)
        filter_coefs = torch.exp(torch.divide(-1.0 * (T1_tensor - T2_tensor) ** 2, 2.0 * (sigma_tensor ** 2)))
        filter_coefs = filter_coefs / torch.sum(filter_coefs, dim=1, keepdim=True)
        # The perturbation is obtained by replacing each input by the linear combination weighted by Gaussian coefs
        X_pert = torch.einsum("asti,si->ati", filter_coefs, X)
        return X_pert


class FadeMovingAverageWindow(Perturbation):
    """This class allows to create and apply 'fade to moving average' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        window_size: Size of the window where each moving average is computed (called W in the paper).
    """

    def __init__(self, device, window_size=2, eps=1.0e-7):
        super().__init__(eps=eps, device=device)
        self.window_size = window_size

    def apply(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        T = X.shape[0]
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients of the perturbation tensor
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = torch.abs(T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs, X)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + mask_tensor * (X - X_avg)
        return X_pert

    def apply_extremal(self, X: torch.Tensor, masks_tensor: torch.Tensor):
        N_area, T, N_features = masks_tensor.shape
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients for the Gaussian perturbation
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = torch.abs(T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs, X[0, :, :])
        X_avg = X_avg.unsqueeze(0)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + masks_tensor * (X - X_avg)
        return X_pert


class FadeMovingAveragePastWindow(Perturbation):
    """This class allows to create and apply 'fade to past moving average' perturbations on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        window_size: Size of the window where each moving average is computed (called W in the paper).
    """

    def __init__(self, device, window_size=2, eps=1.0e-7):
        super().__init__(eps=eps, device=device)
        self.window_size = window_size

    def apply(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        T = X.shape[0]
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients of the perturbation tensor
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = (T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs, X)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + mask_tensor * (X - X_avg)
        return X_pert

    def apply_extremal(self, X: torch.Tensor, masks_tensor: torch.Tensor):
        N_area, T, N_features = masks_tensor.shape
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        T_axis = torch.arange(1, T + 1, dtype=int, device=self.device)
        # For each feature and each time, we compute the coefficients of the perturbation tensor
        T1_tensor = T_axis.unsqueeze(1)
        T2_tensor = T_axis.unsqueeze(0)
        filter_coefs = (T1_tensor - T2_tensor) <= self.window_size
        filter_coefs = filter_coefs / (2 * self.window_size + 1)
        X_avg = torch.einsum("st,si->ti", filter_coefs, X[0, :, :])
        X_avg = X_avg.unsqueeze(0)
        # The perturbation is just an affine combination of the input and the previous tensor weighted by the mask
        X_pert = X_avg + masks_tensor * (X - X_avg)
        return X_pert


class FadeReference(Perturbation):
    """This class allows to create and apply 'fade to reference' perturbation on inputs based on masks.

    Attributes:
        mask_tensor (torch.tensor): The mask tensor than indicates the intensity of the perturbation
            to be applied at each input entry.
        eps (float): Small number used for numerical stability.
        device: Device on which the tensor operations are executed.
        X_ref: The baseline input of same size as X.
    """

    def __init__(self, device, X_ref, eps=1.0e-7):
        super().__init__(eps=eps, device=device)
        self.X_ref = X_ref

    def apply(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        # The perturbation is just an affine combination of the input and the baseline weighted by the mask
        X_pert = self.X_ref + mask_tensor * (X - self.X_ref)
        return X_pert

    def apply_extremal(self, X, mask_tensor):
        super().apply(X=X, mask_tensor=mask_tensor)
        # The perturbation is just an affine combination of the input and the baseline weighted by the mask
        X_pert = self.X_ref + mask_tensor * (X - self.X_ref)
        return X_pert
