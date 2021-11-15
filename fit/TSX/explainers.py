import os
import re
import sys

import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import DeepLift, GradientShap, IntegratedGradients
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tnrange, tqdm_notebook

from fit.TSX.generator import train_joint_feature_generator

# from TSX.generator import JointFeatureGenerator, train_joint_feature_generator, JointDistributionGenerator
from fit.TSX.utils import AverageMeter

eps = 1e-10


def kl_multiclass(p1, p2, reduction="none"):
    return torch.nn.KLDivLoss(reduction=reduction)(torch.log(p2 + eps), p1 + eps)


def kl_multilabel(p1, p2, reduction="none"):
    # treats each column as separate class and calculates KL over the class, sums it up and sends batched
    n_classes = p1.shape[1]
    total_kl = torch.zeros(p1.shape)
    for n in range(n_classes):
        p2_tensor = torch.stack([p2[:, n], 1 - p2[:, n]], dim=1)
        p1_tensor = torch.stack([p1[:, n], 1 - p1[:, n]], dim=1)
        kl = torch.nn.KLDivLoss(reduction=reduction)(torch.log(p2_tensor), p1_tensor)
        total_kl[:, n] = torch.sum(kl, dim=1)
    return total_kl


class FITExplainer:
    def __init__(self, model, generator=None, activation=torch.nn.Softmax(-1), n_classes=2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = generator
        self.base_model = model.to(self.device)
        self.activation = activation
        self.n_classes = n_classes

    def fit_generator(self, generator_model, train_loader, test_loader, n_epochs=300, cv=0):
        train_joint_feature_generator(
            generator_model,
            train_loader,
            test_loader,
            generator_type="joint_generator",
            n_epochs=300,
            lr=0.001,
            weight_decay=0,
            cv=cv,
        )
        self.generator = generator_model.to(self.device)

    def attribute(self, x, y, n_samples=10, retrospective=False, distance_metric="kl", subsets=None):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        self.generator.eval()
        self.generator.to(self.device)
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(list(x.shape))
        if retrospective:
            p_y_t = self.activation(self.base_model(x))

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.activation(self.base_model(x[:, :, : t + 1]))
                p_tm1 = self.activation(self.base_model(x[:, :, 0:t]))

            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                div_all = []
                t1_all = []
                t2_all = []
                for _ in range(n_samples):
                    x_hat_t, _ = self.generator.forward_conditional(x[:, :, :t], x[:, :, t], [i])
                    x_hat[:, :, t] = x_hat_t
                    y_hat_t = self.activation(self.base_model(x_hat))
                    if distance_metric == "kl":
                        if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                            div = torch.sum(
                                torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1
                            ) - torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t), -1)
                            lhs = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1)
                            rhs = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t), -1)
                            # div = torch.where(rhs>lhs, torch.zeros(rhs.shape), rhs/lhs)
                        else:
                            t1 = kl_multilabel(p_y_t, p_tm1)
                            t2 = kl_multilabel(p_y_t, y_hat_t)
                            div, _ = torch.max(t1 - t2, dim=1)
                            # div = div[:,0] #flatten
                        div_all.append(div.cpu().detach().numpy())
                    elif distance_metric == "mean_divergence":
                        div = torch.abs(y_hat_t - p_y_t)
                        div_all.append(np.mean(div.detach().cpu().numpy(), -1))
                    elif distance_metric == "LHS":
                        div = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1)
                        div_all.append(div.cpu().detach().numpy())
                    elif distance_metric == "RHS":
                        div = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t), -1)
                        div_all.append(div.cpu().detach().numpy())
                E_div = np.mean(np.array(div_all), axis=0)
                if distance_metric == "kl":
                    # score[:, i, t] = E_div
                    score[:, i, t] = 2.0 / (1 + np.exp(-5 * E_div)) - 1
                elif distance_metric == "mean_divergence":
                    score[:, i, t] = 1 - E_div
                else:
                    score[:, i, t] = E_div
        return score


class FFCExplainer:
    def __init__(self, model, generator=None, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = generator
        self.base_model = model.to(self.device)
        self.activation = activation

    def fit_generator(self, generator_model, train_loader, test_loader, n_epochs=300):
        train_joint_feature_generator(
            generator_model, train_loader, test_loader, generator_type="joint_generator", n_epochs=n_epochs
        )
        self.generator = generator_model.to(self.device)

    def attribute(self, x, y, n_samples=10, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        self.generator.eval()
        self.generator.to(self.device)
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.activation(self.base_model(x))
        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.activation(self.base_model(x[:, :, : min((t + 1), t_len)]))
            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(n_samples):
                    x_hat_t = self.generator.forward_joint(x[:, :, :t])
                    x_hat[:, i, t] = x_hat_t[:, i]
                    y_hat_t = self.activation(self.base_model(x_hat))
                    if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                        # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                        kl = kl_multiclass(p_y_t, y_hat_t)
                    else:
                        # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                        kl = kl_multilabel(p_y_t, y_hat_t)
                    kl_all.append(torch.sum(kl, -1).detach().cpu().numpy())
                E_kl = np.mean(np.array(kl_all), axis=0)
                score[:, i, t] = E_kl  # * 1e-6
        return score


class FOExplainer:
    def __init__(self, model, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        self.activation = activation

    def attribute(self, x, y, retrospective=False, n_samples=10):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.activation(self.base_model(x))
        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.activation(self.base_model(x[:, :, : t + 1]))
            for i in range(n_features):
                x_hat = x[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(n_samples):
                    x_hat[:, i, t] = torch.Tensor(
                        np.random.uniform(-3, +3, size=(len(x),))
                    )  # torch.Tensor(np.array([np.random.uniform(-3,+3)]).reshape(-1)).to(self.device)
                    y_hat_t = self.activation(self.base_model(x_hat))
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs(y_hat_t - p_y_t)
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i, t] = E_kl
        return score


class AFOExplainer:
    def __init__(self, model, train_loader, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])
        self.activation = activation

    def attribute(self, x, y, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = self.activation(self.base_model(x))

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = self.activation(self.base_model(x[:, :, : t + 1]))
            for i in range(n_features):
                feature_dist = np.array(self.data_distribution[:, i, :]).reshape(-1)
                x_hat = x[:, :, 0 : t + 1].clone()
                kl_all = []
                for _ in range(10):
                    x_hat[:, i, t] = torch.Tensor(np.random.choice(feature_dist, size=(len(x),))).to(self.device)
                    y_hat_t = self.activation(self.base_model(x_hat))
                    # kl = torch.nn.KLDivLoss(reduction='none')(torch.log(y_hat_t), p_y_t)
                    kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                    # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
                    kl_all.append(np.mean(kl.detach().cpu().numpy(), -1))
                E_kl = np.mean(np.array(kl_all), axis=0)
                # score[:, i, t] = 2./(1+np.exp(-1*E_kl)) - 1.
                score[:, i, t] = E_kl
        return score


class MeanImpExplainer:
    def __init__(self, model, train_loader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        trainset = list(train_loader.dataset)
        self.data_distribution = torch.stack([x[0] for x in trainset])

    def attribute(self, x, y, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = torch.nn.Softmax(-1)(self.base_model(x))

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = torch.nn.Softmax(-1)(self.base_model(x[:, :, : t + 1]))
            for i in range(n_features):
                p_tm1 = torch.nn.Softmax(-1)(self.base_model(x[:, :, 0:t]))
                feature_dist = np.array(self.data_distribution[:, i, :]).reshape(-1)
                x_hat = x[:, :, 0 : t + 1].clone()
                x_hat[:, i, t] = (torch.zeros(size=(len(x),)) + np.mean(feature_dist)).to(self.device)
                y_hat_t = torch.nn.Softmax(-1)(self.base_model(x_hat))
                kl = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1) - torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t), -1
                )
                # kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                score[:, i, t] = np.mean(kl.detach().cpu().numpy(), -1)
        return score


class CarryForwardExplainer:
    def __init__(self, model, train_loader):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        trainset = list(train_loader.dataset)

    def attribute(self, x, y, retrospective=False):
        """
        Compute importance score for a sample x, over time and features
        :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
        :param n_samples: number of Monte-Carlo samples
        :return: Importance score matrix of shape:[batch, features, time]
        """
        x = x.to(self.device)
        _, n_features, t_len = x.shape
        score = np.zeros(x.shape)
        if retrospective:
            p_y_t = torch.nn.Softmax(-1)(self.base_model(x))

        for t in range(1, t_len):
            if not retrospective:
                p_y_t = torch.nn.Softmax(-1)(self.base_model(x[:, :, : t + 1]))
            for i in range(n_features):
                p_tm1 = torch.nn.Softmax(-1)(self.base_model(x[:, :, 0:t]))
                x_hat = x[:, :, 0 : t + 1].clone()
                x_hat[:, i, t] = x_hat[:, i, t - 1].to(self.device)
                y_hat_t = torch.nn.Softmax(-1)(self.base_model(x_hat))
                kl = torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1), p_y_t), -1) - torch.sum(
                    torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t), p_y_t), -1
                )
                # kl = torch.abs((y_hat_t[:, :]) - (p_y_t[:, :]))
                score[:, i, t] = np.mean(kl.detach().cpu().numpy(), -1)
        return score


class RETAINexplainer:
    def __init__(self, model, data):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        self.data = data

    def _epoch(self, loader, criterion, optimizer=None, train=False):
        if train and not optimizer:
            raise AttributeError("Optimizer should be given for training")

        if train:
            self.base_model.train()
            mode = "Train"
        else:
            self.base_model.eval()
            mode = "Eval"

        losses = AverageMeter()
        labels = []
        outputs = []

        for bi, batch in enumerate(tqdm_notebook(loader, desc="{} batches".format(mode), leave=False)):
            inputs, targets = batch
            lengths = torch.randint(low=4, high=inputs.shape[2], size=(len(inputs),))
            lengths, _ = torch.sort(lengths, descending=True)
            lengths[0] = inputs.shape[-1]
            inputs = inputs.permute(0, 2, 1)  # Shape: (batch, length, features)
            if self.data == "mimic_int":
                # this is multilabel with labels over time
                targets = targets[torch.range(0, len(inputs) - 1).long(), :, lengths - 1]
                targets = torch.argmax(targets, dim=1)
            elif (
                self.data == "simulation"
                or self.data == "simulation_spike"
                or self.data == "simulation_l2x"
                or self.data == "state"
            ):
                targets = targets[torch.range(0, len(inputs) - 1).long(), lengths - 1]
            elif self.data == "mimic":  # does not have labels over time
                targets = targets[torch.range(0, len(inputs) - 1).long()]

            input_var = torch.autograd.Variable(inputs)
            target_var = torch.autograd.Variable(targets)
            input_var = input_var.to(self.device)
            target_var = target_var.to(self.device)

            output, alpha, beta = self.base_model(input_var, lengths)
            loss = criterion(output, target_var.long())

            labels.append(targets)

            # since the outputs are logit, not probabilities
            outputs.append(torch.nn.functional.softmax(output).data)

            # record loss
            losses.update(loss.item(), inputs.size(0))

            # compute gradient and do update step
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return torch.cat(labels, 0), torch.cat(outputs, 0), losses.avg

    def _one_hot(self, targets, n_classes=4):
        targets_onehot = torch.FloatTensor(targets.shape[0], n_classes)  # .to(self.device)
        targets_onehot.zero_()
        targets_onehot.scatter_(1, targets.view(-1, 1), 1)
        return targets_onehot

    def fit_model(self, train_loader, valid_loader, test_loader, epochs=10, lr=0.001, plot=False, cv=0):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(self.base_model.parameters(), lr=lr*10, momentum=0.95)

        best_valid_epoch = 0
        best_valid_loss = sys.float_info.max
        best_valid_auc = 0.0
        best_valid_aupr = 0.0

        train_losses = []
        valid_losses = []

        if plot:
            # initialise the graph and settings
            fig = plt.figure(figsize=(12, 9))  # , facecolor='w', edgecolor='k')
            ax = fig.add_subplot(111)
            plt.ion()
            fig.show()
            fig.canvas.draw()

        for ei in tnrange(epochs, desc="Epochs"):
            # Train
            train_y_true, train_y_pred, train_loss = self._epoch(
                train_loader, criterion=criterion, optimizer=optimizer, train=True
            )
            if self.data == "mimic_int":
                train_y_true = self._one_hot(train_y_true)
            train_losses.append(train_loss)

            # Eval
            valid_y_true, valid_y_pred, valid_loss = self._epoch(valid_loader, criterion=criterion)
            if self.data == "mimic_int":
                valid_y_true = self._one_hot(valid_y_true)
            valid_losses.append(valid_loss)

            print("Epoch {} - Loss train: {}, valid: {}".format(ei, train_loss, valid_loss))

            valid_y_true.to(self.device)
            valid_y_pred.to(self.device)

            if self.data == "mimic_int":
                valid_auc = roc_auc_score(valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy(), average="weighted")
                valid_aupr = average_precision_score(
                    valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy(), average="weighted"
                )
            else:
                valid_auc = roc_auc_score(
                    valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy()[:, 1], average="weighted"
                )
                valid_aupr = average_precision_score(
                    valid_y_true.cpu().numpy(), valid_y_pred.cpu().numpy()[:, 1], average="weighted"
                )

            is_best = valid_auc > best_valid_auc

            if is_best:
                best_valid_epoch = ei
                best_valid_loss = valid_loss
                best_valid_auc = valid_auc
                best_valid_aupr = valid_aupr

                # evaluate on the test set
                test_y_true, test_y_pred, test_loss = self._epoch(test_loader, criterion=criterion)
                if self.data == "mimic_int":
                    test_y_true = self._one_hot(test_y_true)

                train_y_true.to(self.device)
                train_y_pred.to(self.device)
                test_y_true.to(self.device)
                test_y_pred.to(self.device)

                if self.data == "mimic_int":
                    train_auc = roc_auc_score(
                        train_y_true.cpu().numpy(), train_y_pred.cpu().numpy(), average="weighted"
                    )
                    train_aupr = average_precision_score(
                        train_y_true.cpu().numpy(), train_y_pred.cpu().numpy(), average="weighted"
                    )
                    test_auc = roc_auc_score(test_y_true.cpu().numpy(), test_y_pred.cpu().numpy(), average="weighted")
                    test_aupr = average_precision_score(
                        test_y_true.cpu().numpy(), test_y_pred.cpu().numpy(), average="weighted"
                    )
                else:
                    train_auc = roc_auc_score(
                        train_y_true.cpu().numpy(), train_y_pred.cpu().numpy()[:, 1], average="weighted"
                    )
                    train_aupr = average_precision_score(
                        train_y_true.cpu().numpy(), train_y_pred.cpu().numpy()[:, 1], average="weighted"
                    )
                    test_auc = roc_auc_score(
                        test_y_true.cpu().numpy(), test_y_pred.cpu().numpy()[:, 1], average="weighted"
                    )
                    test_aupr = average_precision_score(
                        test_y_true.cpu().numpy(), test_y_pred.cpu().numpy()[:, 1], average="weighted"
                    )

                if not os.path.exists("./experiments/results/%s" % self.data):
                    os.mkdir("./experiments/results/%s" % self.data)
                torch.save(self.base_model.state_dict(), "./experiments/results/%s/retain_%d.pt" % (self.data, cv))

            # plot
            if plot:
                ax.clear()
                ax.plot(np.arange(len(train_losses)), np.array(train_losses), label="Training Loss")
                ax.plot(np.arange(len(valid_losses)), np.array(valid_losses), label="Validation Loss")
                ax.set_xlabel("epoch")
                ax.set_ylabel("Loss")
                ax.legend(loc="best")
                plt.tight_layout()
                plt.savefig(os.path.join("./plots", self.data, "retain_train_loss.pdf"))
                # fig.canvas.draw()

        print("Best Validation Epoch: {}\n".format(best_valid_epoch))
        print("Best Validation Loss: {}\n".format(best_valid_loss))
        print("Best Validation AUROC: {}\n".format(best_valid_auc))
        print("Best Validation AUPR: {}\n".format(best_valid_aupr))
        print("Test Loss: {}\n".format(test_loss))
        print("Test AUROC: {}\n".format(test_auc))
        print("Test AUPR: {}\n".format(test_aupr))

    def attribute(self, x, y):
        score = np.zeros(x.shape)
        x = x.permute(0, 2, 1)  # shape:[batch, time, feature]
        logit, alpha, beta = self.base_model(x, (torch.ones((len(x),)) * x.shape[1]).long())
        w_emb = self.base_model.embedding[1].weight
        for i in range(x.shape[2]):
            for t in range(x.shape[1]):
                imp = self.base_model.output(beta[:, t, :] * w_emb[:, i].expand_as(beta[:, t, :]))
                score[:, i, t] = (
                    (alpha[:, t, 0] * imp[torch.range(0, len(imp) - 1).long(), y.long()] * x[:, t, i])
                    .detach()
                    .cpu()
                    .numpy()
                )
        return score


class DeepLiftExplainer:
    def __init__(self, model, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        self.explainer = DeepLift(self.base_model)
        self.activation = activation

    def attribute(self, x, y, retrospective=False):
        self.base_model.zero_grad()
        if retrospective:
            score = self.explainer.attribute(x, target=y.long(), baselines=(x * 0))
            score = abs(score.detach().cpu().numpy())
        else:
            score = np.zeros(x.shape)
            for t in range(1, x.shape[-1]):
                x_in = x[:, :, : t + 1]
                pred = self.activation(self.base_model(x_in))
                if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                    target = torch.argmax(pred, -1)
                    imp = self.explainer.attribute(x_in, target=target.long(), baselines=(x[:, :, : t + 1] * 0))
                    score[:, :, t] = abs(imp.detach().cpu().numpy()[:, :, -1])
                else:
                    # this works for multilabel and single prediction aka spike
                    n_labels = pred.shape[1]
                    if n_labels > 1:
                        imp = torch.zeros(list(x_in.shape) + [n_labels])
                        for l in range(n_labels):  # noqa: E741
                            target = (pred[:, l] > 0.5).float()  # [:,0]
                            imp[:, :, :, l] = self.explainer.attribute(x_in, target=target.long(), baselines=(x_in * 0))
                        score[:, :, t] = (imp.detach().cpu().numpy()).max(3)[:, :, -1]
                    else:
                        # this is for spike with just one label. and we will explain one cla
                        target = (pred > 0.5).float()[:, 0]
                        imp = self.explainer.attribute(x_in, target=target.long(), baselines=(x[:, :, : t + 1] * 0))
                        score[:, :, t] = abs(imp.detach().cpu().numpy()[:, :, -1])
        return score


class IGExplainer:
    def __init__(self, model, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        self.base_model.device = self.device
        self.explainer = IntegratedGradients(self.base_model)
        self.activation = activation

    def attribute(self, x, y, retrospective=False):
        # x, y = x.to(self.device), y.to(self.device)
        score = np.zeros(x.shape)
        self.base_model.zero_grad()
        if retrospective:
            score = self.attribute(x, target=y.long(), baselines=(x * 0))
            score = score.detach().cpu().numpy()
        else:
            score = np.zeros(x.shape)
            for t in range(x.shape[-1]):
                x_in = x[:, :, : t + 1]
                pred = self.activation(self.base_model(x_in))
                if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                    target = torch.argmax(pred, -1)
                    imp = self.explainer.attribute(x_in, target=target, baselines=(x[:, :, : t + 1] * 0))
                    score[:, :, t] = imp.detach().cpu().numpy()[:, :, -1]
                else:
                    # print(pred)
                    n_labels = pred.shape[1]
                    if n_labels > 1:
                        imp = torch.zeros(list(x_in.shape) + [n_labels])
                        for l in range(n_labels):  # noqa: E741
                            target = (pred[:, l] > 0.5).float()  # [:,0]
                            imp[:, :, :, l] = self.explainer.attribute(x_in, target=target.long(), baselines=(x_in * 0))
                        score[:, :, t] = (imp.detach().cpu().numpy()).max(3)[:, :, -1]
                    else:
                        # this is for spike with just one label. and we will explain one class
                        target = (pred > 0.5).float()[:, 0]
                        imp = self.explainer.attribute(x_in, target=target.long(), baselines=(x_in * 0))
                        score[:, :, t] = imp.detach().cpu().numpy()[:, :, -1]
        return score


class GradientShapExplainer:
    def __init__(self, model, activation=torch.nn.Softmax(-1)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model.to(self.device)
        self.base_model.device = self.device
        self.explainer = GradientShap(self.base_model)
        self.activation = activation

    def attribute(self, x, y, retrospective=False):
        x, y = x.to(self.device), y.to(self.device)
        if retrospective:
            score = self.explainer.attribute(
                x, target=y.long(), n_samples=50, stdevs=0.0001, baselines=torch.cat([x * 0, x * 1])
            )
            score = abs(score.cpu().numpy())
        else:
            score = np.zeros(x.shape)

            for t in range(x.shape[-1]):
                x_in = x[:, :, : t + 1]
                pred = self.activation(self.base_model(x_in))
                if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                    target = torch.argmax(pred, -1)
                    imp = self.explainer.attribute(
                        x_in,
                        target=target.long(),
                        n_samples=50,
                        stdevs=0.0001,
                        baselines=torch.cat([x[:, :, : t + 1] * 0, x[:, :, : t + 1] * 1]),
                    )
                    score[:, :, t] = imp.cpu().numpy()[:, :, -1]
                else:
                    n_labels = pred.shape[1]
                    if n_labels > 1:
                        imp = torch.zeros(list(x_in.shape) + [n_labels])
                        for l in range(n_labels):  # noqa: E741
                            target = (pred[:, l] > 0.5).float()  # [:,0]
                            imp[:, :, :, l] = self.explainer.attribute(x_in, target=target.long(), baselines=(x_in * 0))
                        score[:, :, t] = (imp.detach().cpu().numpy()).max(3)[:, :, -1]
                    else:
                        # this is for spike with just one label. and we will explain one cla
                        target = (pred > 0.5).float()[:, 0]
                        imp = self.explainer.attribute(x_in, target=target.long(), baselines=(x[:, :, : t + 1] * 0))
                        score[:, :, t] = abs(imp.detach().cpu().numpy()[:, :, -1])
        return score


# class SHAPExplainer:
#     def __init__(self, model, train_loader):
#         self.device = "cuda"  # 'cuda' if torch.cuda.is_available() else 'cpu'
#         model.to(self.device)
#         self.base_model = model
#         self.base_model.device = self.device
#         trainset = list(train_loader.dataset)
#         x_train = torch.stack([x[0] for x in trainset])
#         background = x_train[np.random.choice(np.arange(len(x_train)), 100, replace=False)]
#         self.explainer = shap.DeepExplainer(self.base_model, background.to(self.device))

#     def attribute(self, x, y, retrospective=False):
#         x.to(self.device)
#         if retrospective:
#             score = self.explainer.shap_values(x)
#         else:
#             score = np.zeros(x.shape)
#             for t in range(1, x.shape[-1]):
#                 imp = self.explainer.shap_values(torch.reshape(x[:, :, : t + 1], (x.shape[0], -1)))
#                 score[:, :, t] = imp.detach().cpu().numpy()[:, :, -1]
#         return score


class LIMExplainer:
    def __init__(self, model, train_loader, activation=torch.nn.Softmax(-1), n_classes=2):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model = model
        self.base_model.device = self.device
        trainset = list(train_loader.dataset)
        self.activation = activation
        x_train = torch.stack([x[0] for x in trainset]).to(self.device)
        x_train = x_train[torch.arange(len(x_train)), :, torch.randint(5, x_train.shape[-1], (len(x_train),))]
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            x_train.cpu().numpy(),
            feature_names=["f%d" % c for c in range(x_train.shape[1])],
            discretize_continuous=True,
        )
        self.n_classes = n_classes

    def _predictor_wrapper(self, sample):
        """
        In order to use the lime explainer library we need to go back and forth between numpy library (compatible with Lime)
        and torch (Compatible with the predictor model). This wrapper helps with this
        :param sample: input sample for the predictor (type: numpy array)
        :return: one-hot model output (type: numpy array)
        """
        torch_in = torch.Tensor(sample).reshape(len(sample), -1, 1)
        torch_in.to(self.device)
        out = self.base_model(torch_in)
        out = self.activation(out)
        return out.detach().cpu().numpy()

    def attribute(self, x, y, retrospective=False):
        x = x.cpu().numpy()
        score = np.zeros(x.shape)
        for sample_ind, sample in enumerate(x):
            for t in range(1, x.shape[-1]):
                imp = self.explainer.explain_instance(sample[:, t], self._predictor_wrapper, top_labels=self.n_classes)
                # This likely should change
                if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
                    for ind, st in enumerate(imp.as_list()):
                        imp_score = st[1]
                        terms = re.split("< | > | <= | >=", st[0])
                        for feat in range(x.shape[1]):
                            if "f%d" % feat in terms:
                                score[sample_ind, feat, t] = imp_score
                else:
                    for k in imp.local_exp.keys():
                        imp_score = imp.local_exp[k]
                        for term in imp_score:
                            score[sample_ind, term[0], t] += term[1]

            print("sample:", sample_ind, " done")
        return score


# class FITSubGroupExplainer:
#     def __init__(self, model, generator=None, activation=torch.nn.Softmax(-1)):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.generator = generator
#         self.base_model = model.to(self.device)
#         self.activation = activation

#     def fit_generator(self, generator_model, train_loader, test_loader, n_epochs=300):
#         train_joint_feature_generator(
#             generator_model,
#             train_loader,
#             test_loader,
#             generator_type="joint_generator",
#             n_epochs=300,
#             lr=0.001,
#             weight_decay=0,
#         )
#         self.generator = generator_model.to(self.device)

#     def attribute(self, x, y, n_samples=10, retrospective=False, distance_metric="kl"):
#         """
#         Compute importance score for a sample x, over time and features
#         :param x: Sample instance to evaluate score for. Shape:[batch, features, time]
#         :param n_samples: number of Monte-Carlo samples
#         :return: Importance score matrix of shape:[batch, features, time]
#         """
#         self.generator.eval()
#         self.generator.to(self.device)
#         x = x.to(self.device)
#         _, n_features, t_len = x.shape
#         score = np.zeros(x.shape)
#         if retrospective:
#             p_y_t = self.activation(self.base_model(x))

#         for t in range(1, t_len):
#             if not retrospective:
#                 p_y_t = self.activation(self.base_model(x[:, :, : t + 1]))
#             x_hat = x[:, :, 0 : t + 1].clone()
#             div_all = []
#             p_tm1 = self.activation(self.base_model(x[:, :, 0:t]))
#             for _ in range(n_samples):
#                 x_hat_t, _ = self.generator.forward_conditional(x[:, :, :t], x[:, :, t], S)
#                 x_hat[:, :, t] = x_hat_t
#                 y_hat_t = self.activation(self.base_model(x_hat))
#                 if distance_metric == "kl":
#                     # kl = torch.nn.KLDivLoss(reduction='none')(torch.Tensor(np.log(y_hat_t)).to(self.device), p_y_t)
#                     if type(self.activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # noqa: E721
#                         div = torch.sum(
#                             torch.nn.KLDivLoss(reduction="none")(torch.log(p_tm1 + eps), p_y_t + eps), -1
#                         ) - torch.sum(torch.nn.KLDivLoss(reduction="none")(torch.log(y_hat_t + eps), p_y_t + eps), -1)
#                     else:
#                         div = torch.sum(kl_multilabel(p_y_t, p_tm1), -1) - torch.sum(kl_multilabel(p_y_t, y_hat_t), -1)
#                     # print(x_hat_t[0], x[0, :, t], self.generator.forward_joint(x[:, :, :t])[0])
#                     # kl_all.append(torch.sum(kl, -1).cpu().detach().numpy())
#                     div_all.append(div.cpu().detach().numpy())
#                 elif distance_metric == "mean_divergence":
#                     div = torch.abs(y_hat_t - p_y_t)
#                     div_all.append(np.mean(div.detach().cpu().numpy(), -1))
#             E_div = np.mean(np.array(div_all), axis=0)
#             if distance_metric == "kl":
#                 # score[:, i, t] = 2./(1+np.exp(-1*E_div)) - 1
#                 score[:, i, t] = E_div
#             elif distance_metric == "mean_divergence":
#                 score[:, i, t] = 1 - E_div
#         return score
