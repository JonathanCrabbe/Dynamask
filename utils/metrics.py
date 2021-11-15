import torch
import numpy as np
from utils.tensor_manipulation import extract_subtensor


def get_information(
    saliency: torch.Tensor, ids_time=None, ids_feature=None, normalize: bool = False, eps: float = 1.0e-5
):
    """
    This methods returns the information contained in the identifiers for the saliency tensor.
    :param normalize: True if the information should be normalized
    :param eps: small parameter for numerical stability
    :param saliency: the (T, N_features) saliency tensor from which information is extracted
    :param ids_time: list of the times that should contribute
    :param ids_feature: list of the features that should contribute
    :return: information content as a torch scalar
    """
    subsaliency = extract_subtensor(saliency, ids_time, ids_feature)
    subsaliency_information = (torch.abs(torch.log2(1 - subsaliency + eps))).sum()
    if normalize:
        saliency_information = (torch.abs(torch.log2(1 - subsaliency + eps))).sum()
        subsaliency_information /= saliency_information
    return subsaliency_information.cpu().item()


def get_entropy(saliency: torch.Tensor, ids_time=None, ids_feature=None, normalize: bool = False, eps: float = 1.0e-5):
    """
    This methods returns the entropy contained in the identifiers for the saliency tensor.
    :param saliency: the (T, N_features) saliency tensor from which information is extracted
    :param normalize: True if the entropy should be normalized
    :param eps: small parameter for numerical stability
    :param ids_time: list of the times that should contribute
    :param ids_feature: list of the features that should contribute
    :return: entropy as a torch scalar
    """
    subsaliency = extract_subtensor(saliency, ids_time, ids_feature)
    subentropy_tensor = subsaliency * torch.abs(torch.log2(eps + subsaliency)) + (1 - subsaliency) * torch.abs(
        torch.log2(eps + 1 - subsaliency)
    )
    subsaliency_entropy = subentropy_tensor.sum()
    if normalize:
        entropy_tensor = saliency * torch.abs(torch.log2(eps + saliency)) + (1 - saliency) * torch.abs(
            torch.log2(eps + 1 - saliency)
        )

        saliency_entropy = entropy_tensor.sum()
        subsaliency_entropy /= saliency_entropy
    return subsaliency_entropy.cpu().item()


def get_information_array(saliency: np.ndarray, eps: float = 1.0e-5):
    """
    This methods returns the information contained in the identifiers for the saliency numpy array.
    :param eps: small parameter for numerical stability
    :param saliency: the (N_samples) saliency array from which information is extracted
    :return: information content as a torch scalar
    """
    saliency_information = (np.abs(np.log2(1 - saliency + eps))).sum()
    return saliency_information


def get_entropy_array(saliency: np.ndarray, eps: float = 1.0e-5):
    """
    This methods returns the entropy contained in the identifiers for the saliency numpy array.
    :param saliency: the (N_samples) saliency array from which information is extracted
    :param eps: small parameter for numerical stability
    :return: entropy as a torch scalar
    """
    entropy_tensor = saliency * np.abs(np.log2(eps + saliency)) + (1 - saliency) * np.abs(np.log2(eps + 1 - saliency))
    saliency_entropy = entropy_tensor.sum()
    return saliency_entropy
