import torch


def cross_entropy(proba_pred, proba_target):
    """Computes the cross entropy between the two probabilities torch tensors."""
    return -(proba_target * torch.log(proba_pred)).mean()


def log_loss(proba_pred, proba_target):
    """Computes the log loss between the two probabilities torch tensors."""
    label_target = torch.argmax(proba_target, dim=-1, keepdim=True)
    proba_select = torch.gather(proba_pred, -1, label_target)
    return -(torch.log(proba_select)).mean()


def log_loss_target(proba_pred, target):
    """Computes log loss between the target and the predicted probabilities expressed as torch tensors.

    The target is a one dimensional tensor whose dimension matches the first dimension of proba_pred.
    It contains integers that represent the true class for each instance.
    """
    proba_select = torch.gather(proba_pred, -1, target)
    return -(torch.log(proba_select)).mean()


def mse(Y, Y_target):
    """Computes the mean squared error between Y and Y_target."""
    return torch.mean((Y - Y_target) ** 2)
