import argparse
import os
import pickle as pkl
import time

import numpy as np
import seaborn as sns
import torch
from matplotlib import rc
from sklearn import metrics

from fit.TSX.explainers import (  # SHAPExplainer,
    AFOExplainer,
    CarryForwardExplainer,
    DeepLiftExplainer,
    FFCExplainer,
    FITExplainer,
    FOExplainer,
    GradientShapExplainer,
    IGExplainer,
    LIMExplainer,
    MeanImpExplainer,
    RETAINexplainer,
)
from fit.TSX.generator import JointDistributionGenerator, JointFeatureGenerator
from fit.TSX.models import RETAIN, EncoderRNN, StateClassifier, StateClassifierMIMIC
from fit.TSX.utils import (
    compute_median_rank,
    load_data,
    load_simulated_data,
    train_model,
    train_model_multiclass,
    train_model_rt,
    train_model_rt_binary,
)

sns.set()
rc("font", weight="bold")

intervention_list = [
    "vent",
    "vaso",
    "adenosine",
    "dobutamine",
    "dopamine",
    "epinephrine",
    "isuprel",
    "milrinone",
    "norepinephrine",
    "phenylephrine",
    "vasopressin",
    "colloid_bolus",
    "crystalloid_bolus",
    "nivdurations",
]
intervention_list_plot = ["niv-vent", "vent", "vaso", "other"]
feature_map_mimic = [
    "ANION GAP",
    "ALBUMIN",
    "BICARBONATE",
    "BILIRUBIN",
    "CREATININE",
    "CHLORIDE",
    "GLUCOSE",
    "HEMATOCRIT",
    "HEMOGLOBIN",
    "LACTATE",
    "MAGNESIUM",
    "PHOSPHATE",
    "PLATELET",
    "POTASSIUM",
    "PTT",
    "INR",
    "PT",
    "SODIUM",
    "BUN",
    "WBC",
    "HeartRate",
    "SysBP",
    "DiasBP",
    "MeanBP",
    "RespRate",
    "SpO2",
    "Glucose",
    "Temp",
]

color_map = [
    "#7b85d4",
    "#f37738",
    "#83c995",
    "#d7369e",
    "#859795",
    "#ad5b50",
    "#7e1e9c",
    "#0343df",
    "#033500",
    "#E0FF66",
    "#4C005C",
    "#191919",
    "#FF0010",
    "#2BCE48",
    "#FFCC99",
    "#808080",
    "#740AFF",
    "#8F7C00",
    "#9DCC00",
    "#F0A3FF",
    "#94FFB5",
    "#FFA405",
    "#FFA8BB",
    "#426600",
    "#005C31",
    "#5EF1F2",
    "#993F00",
    "#990000",
    "#990000",
    "#FFFF80",
    "#FF5005",
    "#FFFF00",
    "#FF0010",
    "#FFCC99",
    "#003380",
]

ks = {"simulation_spike": 1, "simulation": 3, "simulation_l2x": 4}


if __name__ == "__main__":
    np.random.seed(1234)
    parser = argparse.ArgumentParser(description="Run baseline model for explanation")
    parser.add_argument("--explainer", type=str, default="fit", help="Explainer model")
    parser.add_argument("--data", type=str, default="simulation")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--train_gen", action="store_true")
    parser.add_argument("--generator_type", type=str, default="history")
    parser.add_argument("--out_path", type=str, default="./experiments/results/")
    parser.add_argument("--mimic_path", type=str, default="./data/mimic")
    parser.add_argument("--binary", action="store_true", default=False)
    parser.add_argument("--gt", type=str, default="true_model", help="specify ground truth score")
    parser.add_argument("--cv", type=int, default=0, help="cross validation")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 100

    activation = torch.nn.Softmax(-1)
    output_path = args.out_path
    if args.data == "simulation":
        output_path = os.path.join(output_path, "state")
        data_name = "state"
    if args.data == "mimic":
        output_path = os.path.join(output_path, "mimic")
        data_name = "mimic"
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if args.data == "simulation":
        feature_size = 3
        data_path = "./data/state"
        data_type = "state"
        n_classes = 2
    elif args.data == "simulation_l2x":
        feature_size = 3
        data_path = "./data/simulated_data_l2x"
        data_type = "state"
        n_classes = 2
    elif args.data == "simulation_spike":
        feature_size = 3
        data_path = "./data/simulated_spike_data"
        data_type = "spike"
        n_classes = 2  # use with state-classifier
        if args.explainer == "retain":
            activation = torch.nn.Softmax()
        else:
            activation = torch.nn.Sigmoid()
        batch_size = 200
    elif args.data == "mimic":
        data_type = "mimic"
        timeseries_feature_size = len(feature_map_mimic)
        n_classes = 2
        task = "mortality"
    elif args.data == "mimic_int":
        timeseries_feature_size = len(feature_map_mimic)
        data_type = "real"
        n_classes = 4
        batch_size = 256
        task = "intervention"
        # change this to softmax for suresh et al
        activation = torch.nn.Sigmoid()
        # activation = torch.nn.Softmax(-1)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    plot_path = os.path.join("./plots/%s" % args.data)

    # Load data
    if args.data == "mimic" or args.data == "mimic_int":
        if args.mimic_path is None:
            raise ValueError("Specify the data directory containing processed mimic data")
        p_data, train_loader, valid_loader, test_loader = load_data(
            batch_size=batch_size, path=args.mimic_path, task=task, cv=args.cv
        )
        feature_size = p_data.feature_size
        class_weight = p_data.pos_weight
    else:
        _, train_loader, valid_loader, test_loader = load_simulated_data(
            batch_size=batch_size, datapath=data_path, percentage=0.8, data_type=data_type, cv=args.cv
        )

    # Prepare model to explain
    if args.explainer == "retain":
        if args.data == "mimic" or args.data == "simulation" or args.data == "simulation_l2x":
            model = RETAIN(
                dim_input=feature_size,
                dim_emb=128,
                dropout_emb=0.4,
                dim_alpha=8,
                dim_beta=8,
                dropout_context=0.4,
                dim_output=2,
            )
        elif args.data == "mimic_int":
            model = RETAIN(
                dim_input=feature_size,
                dim_emb=32,
                dropout_emb=0.4,
                dim_alpha=16,
                dim_beta=16,
                dropout_context=0.4,
                dim_output=n_classes,
            )
        elif args.data == "simulation_spike":
            model = RETAIN(
                dim_input=feature_size,
                dim_emb=4,
                dropout_emb=0.4,
                dim_alpha=16,
                dim_beta=16,
                dropout_context=0.4,
                dim_output=n_classes,
            )
        explainer = RETAINexplainer(model, data_name)
        if args.train:
            t0 = time.time()
            if args.data == "mimic" or args.data == "simulation" or args.data == "simulation_l2x":
                explainer.fit_model(train_loader, valid_loader, test_loader, lr=1e-3, plot=False, epochs=50)
            else:
                explainer.fit_model(
                    train_loader, valid_loader, test_loader, lr=1e-4, plot=False, epochs=100, cv=args.cv
                )
            print("Total time required to train retain: ", time.time() - t0)
        else:
            model.load_state_dict(
                torch.load(os.path.join("./experiments/results/%s/%s_%d.pt" % (data_name, "retain", args.cv)))
            )
    else:
        if not args.binary:
            if args.data == "mimic_int":
                model = StateClassifierMIMIC(feature_size=feature_size, n_state=n_classes, hidden_size=128, rnn="LSTM")
            else:
                model = StateClassifier(feature_size=feature_size, n_state=n_classes, hidden_size=200, rnn="GRU")
        else:
            model = EncoderRNN(
                feature_size=feature_size, hidden_size=50, regres=True, return_all=False, data=args.data, rnn="GRU"
            )
        if args.train:
            if not args.binary:
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
                if args.data == "mimic":
                    train_model(
                        model,
                        train_loader,
                        valid_loader,
                        optimizer=optimizer,
                        n_epochs=100,
                        device=device,
                        experiment="model",
                        cv=args.cv,
                    )
                elif "simulation" in args.data:
                    train_model_rt(
                        model=model,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        optimizer=optimizer,
                        n_epochs=50,
                        device=device,
                        experiment="model",
                        data="state",
                        cv=args.cv,
                    )
                elif args.data == "mimic_int":
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
                    if type(activation).__name__ == type(torch.nn.Softmax(-1)).__name__:  # suresh et al  # noqa: E721
                        train_model_multiclass(
                            model=model,
                            train_loader=train_loader,
                            valid_loader=test_loader,
                            optimizer=optimizer,
                            n_epochs=50,
                            device=device,
                            experiment="model",
                            data=args.data,
                            num=5,
                            loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).to(device)),
                            cv=args.cv,
                        )
                    else:
                        train_model_multiclass(
                            model=model,
                            train_loader=train_loader,
                            valid_loader=test_loader,
                            optimizer=optimizer,
                            n_epochs=25,
                            device=device,
                            experiment="model",
                            data=args.data,
                            num=5,
                            # loss_criterion=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(class_weight).cuda()),cv=args.cv)
                            loss_criterion=torch.nn.BCEWithLogitsLoss(),
                            cv=args.cv,
                        )
                        # loss_criterion=torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weight).cuda()),cv=args.cv)
                        # loss_criterion=torch.nn.CrossEntropyLoss(),cv=args.cv)
            else:
                # this learning rate works much better for spike data
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                if args.data == "mimic":
                    train_model(
                        model,
                        train_loader,
                        valid_loader,
                        optimizer=optimizer,
                        n_epochs=200,
                        device=device,
                        experiment="model",
                        cv=args.cv,
                    )
                else:
                    train_model_rt_binary(
                        model,
                        train_loader,
                        valid_loader,
                        optimizer=optimizer,
                        n_epochs=250,
                        device=device,
                        experiment="model",
                        data=args.data,
                        cv=args.cv,
                    )

        model.load_state_dict(
            torch.load(os.path.join("./experiments/results/%s/%s_%d.pt" % (data_name, "model", args.cv)))
        )

        if args.explainer == "fit":
            if args.generator_type == "history":
                generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=data_name)
                if args.train:
                    if args.data == "mimic_int" or args.data == "simulation_spike":
                        explainer = FITExplainer(model, activation=torch.nn.Sigmoid(), n_classes=n_classes)
                    else:
                        explainer = FITExplainer(model)
                    explainer.fit_generator(generator, train_loader, valid_loader, cv=args.cv)
                else:
                    generator.load_state_dict(
                        torch.load(
                            os.path.join("./experiments/results/%s/%s_%d.pt" % (data_name, "joint_generator", args.cv))
                        )
                    )
                    if args.data == "mimic_int" or args.data == "simulation_spike":
                        explainer = FITExplainer(model, generator, activation=torch.nn.Sigmoid(), n_classes=n_classes)
                    else:
                        explainer = FITExplainer(model, generator)
            elif args.generator_type == "no_history":
                generator = JointDistributionGenerator(n_components=5, train_loader=train_loader)
                if args.data == "mimic_int" or args.data == "simulation_spike":
                    explainer = FITExplainer(model, generator, activation=torch.nn.Sigmoid())
                else:
                    explainer = FITExplainer(model, generator)

        elif args.explainer == "integrated_gradient":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = IGExplainer(model, activation=activation)
            else:
                explainer = IGExplainer(model)

        elif args.explainer == "deep_lift":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = DeepLiftExplainer(model, activation=activation)
            else:
                explainer = DeepLiftExplainer(model)

        elif args.explainer == "fo":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = FOExplainer(model, activation=activation)
            else:
                explainer = FOExplainer(model)

        elif args.explainer == "afo":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = AFOExplainer(model, train_loader, activation=activation)
            else:
                explainer = AFOExplainer(model, train_loader)

        elif args.explainer == "carry_forward":
            explainer = CarryForwardExplainer(model, train_loader)

        elif args.explainer == "mean_imp":
            explainer = MeanImpExplainer(model, train_loader)

        elif args.explainer == "gradient_shap":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = GradientShapExplainer(model, activation=activation)
            else:
                explainer = GradientShapExplainer(model)

        elif args.explainer == "ffc":
            generator = JointFeatureGenerator(feature_size, hidden_size=feature_size * 3, data=args.data)
            if args.train:
                if args.data == "mimic_int" or args.data == "simulation_spike":
                    explainer = FFCExplainer(model, activation=activation)
                else:
                    explainer = FFCExplainer(model)
                explainer.fit_generator(generator, train_loader, valid_loader)
            else:
                generator.load_state_dict(
                    torch.load(os.path.join("./experiments/results/%s/%s.pt" % (args.data, "joint_generator")))
                )
                if args.data == "mimic_int" or args.data == "simulation_spike":
                    explainer = FFCExplainer(model, generator, activation=activation)
                else:
                    explainer = FFCExplainer(model, generator)

        elif args.explainer == "shap":
            raise NotImplementedError("SHAPExplainer not implemented")

        elif args.explainer == "lime":
            if args.data == "mimic_int" or args.data == "simulation_spike":
                explainer = LIMExplainer(model, train_loader, activation=activation, n_classes=n_classes)
            else:
                explainer = LIMExplainer(model, train_loader)

        elif args.explainer == "retain":
            explainer = RETAINexplainer(model, args.data)
        else:
            raise ValueError("%s explainer not defined!" % args.explainer)

    # Load ground truth for simulations
    if data_type == "state":
        with open(os.path.join(data_path, "state_dataset_importance_test.pkl"), "rb") as f:
            gt_importance_test = pkl.load(f)
        with open(os.path.join(data_path, "state_dataset_states_test.pkl"), "rb") as f:
            state_test = pkl.load(f)
        with open(os.path.join(data_path, "state_dataset_logits_test.pkl"), "rb") as f:
            logits_test = pkl.load(f)
    elif data_type == "spike":
        with open(os.path.join(data_path, "gt_test.pkl"), "rb") as f:
            gt_importance_test = pkl.load(f)

    importance_scores = []
    ranked_feats = []
    n_samples = 1
    for x, y in test_loader:
        model.train()
        model.to(device)
        x = x.to(device)
        y = y.to(device)

        t0 = time.time()
        score = explainer.attribute(x, y if args.data == "mimic" else y[:, -1].long())

        ranked_features = np.array([((-(score[n])).argsort(0).argsort(0) + 1) for n in range(x.shape[0])])
        importance_scores.append(score)
        ranked_feats.append(ranked_features)

    importance_scores = np.concatenate(importance_scores, 0)

    print("Saving file to ", os.path.join(output_path, "%s_test_importance_scores_%d.pkl" % (args.explainer, args.cv)))
    with open(os.path.join(output_path, "%s_test_importance_scores_%d.pkl" % (args.explainer, args.cv)), "wb") as f:
        pkl.dump(importance_scores, f, protocol=pkl.HIGHEST_PROTOCOL)

    ranked_feats = np.concatenate(ranked_feats, 0)
    with open(os.path.join(output_path, "%s_test_ranked_scores.pkl" % args.explainer), "wb") as f:
        pkl.dump(ranked_feats, f, protocol=pkl.HIGHEST_PROTOCOL)

    if "simulation" in args.data:
        gt_soft_score = np.zeros(gt_importance_test.shape)
        gt_importance_test.astype(int)
        gt_score = gt_importance_test.flatten()
        explainer_score = importance_scores.flatten()
        if (
            args.explainer == "deep_lift"
            or args.explainer == "integrated_gradient"
            or args.explainer == "gradient_shap"
        ):
            explainer_score = np.abs(explainer_score)
        auc_score = metrics.roc_auc_score(gt_score, explainer_score)
        aupr_score = metrics.average_precision_score(gt_score, explainer_score)

        _, median_rank, _ = compute_median_rank(ranked_feats, gt_soft_score, soft=True, K=4)
        print("auc:", auc_score, " aupr:", aupr_score)
