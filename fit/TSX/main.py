import argparse
import json
import os
import sys

import numpy as np
from TSX.experiments import (
    Baseline,
    BaselineExplainer,
    EncoderPredictor,
    FeatureGeneratorExplainer,
)
from TSX.utils import load_data, load_ghg_data, load_simulated_data

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

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


MIMIC_TEST_SAMPLES = np.random.randint(1000, size=5)
SIMULATION_SAMPLES = np.random.randint(100, size=10)
samples_to_analyze = {
    "mimic": MIMIC_TEST_SAMPLES,
    "simulation": SIMULATION_SAMPLES,
    "ghg": [],
    "simulation_spike": range(100),
}


def main(experiment, train, data, generator_type, predictor_model, all_samples, cv, output_path):
    print("********** Experiment with the %s data **********" % experiment)
    with open("config.json") as config_file:
        configs = json.load(config_file)[data][experiment]

    if not os.path.exists("./data"):
        os.mkdir("./data")
    ## Load the data
    if data == "mimic":
        p_data, train_loader, valid_loader, test_loader = load_data(
            batch_size=configs["batch_size"], path="./data", cv=cv
        )
        feature_size = p_data.feature_size
    elif data == "ghg":
        p_data, train_loader, valid_loader, test_loader = load_ghg_data(configs["batch_size"], cv=cv)
        feature_size = p_data.feature_size
    elif data == "simulation_spike":
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(
            batch_size=configs["batch_size"], path="./data/simulated_spike_data", data_type="spike", cv=cv
        )
        feature_size = p_data.shape[1]

    elif data == "simulation":
        percentage = 100.0
        p_data, train_loader, valid_loader, test_loader = load_simulated_data(
            batch_size=configs["batch_size"], path="./data/simulated_data", percentage=percentage / 100, cv=cv
        )
        feature_size = p_data.shape[1]

    ## Create the experiment class
    if experiment == "baseline":
        exp = Baseline(train_loader, valid_loader, test_loader, p_data.feature_size)
    elif experiment == "risk_predictor":
        exp = EncoderPredictor(
            train_loader,
            valid_loader,
            test_loader,
            feature_size,
            configs["encoding_size"],
            rnn_type=configs["rnn_type"],
            data=data,
            model=predictor_model,
        )
    elif experiment == "feature_generator_explainer":
        exp = FeatureGeneratorExplainer(
            train_loader,
            valid_loader,
            test_loader,
            feature_size,
            patient_data=p_data,
            output_path=output_path,
            predictor_model=predictor_model,
            generator_hidden_size=configs["encoding_size"],
            prediction_size=1,
            generator_type=generator_type,
            data=data,
            experiment=experiment + "_" + generator_type,
        )
    elif experiment == "lime_explainer":
        exp = BaselineExplainer(
            train_loader, valid_loader, test_loader, feature_size, data_class=p_data, data=data, baseline_method="lime"
        )

    if all_samples:
        print("Experiment on all test data")
        print("Number of test samples: ", len(exp.test_loader.dataset))
        exp.run(
            train=False,
            n_epochs=configs["n_epochs"],
            samples_to_analyze=list(range(0, len(exp.test_loader.dataset))),
            plot=False,
            cv=cv,
        )
    else:
        exp.run(train=train, n_epochs=configs["n_epochs"], samples_to_analyze=samples_to_analyze[data])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ICU mortality prediction model")
    parser.add_argument("--model", type=str, default="feature_generator_explainer", help="Prediction model")
    parser.add_argument("--data", type=str, default="simulation")
    parser.add_argument("--generator", type=str, default="joint_RNN_generator")
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--predictor", type=str, default="RNN")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--all_samples", action="store_true")
    parser.add_argument("--cv", type=int, default=0)
    args = parser.parse_args()
    if args.out == "./out" and not os.path.exists("./out"):
        os.mkdir("./out")
    if not os.path.exists(args.out):
        os.mkdir(args.out)
    main(
        args.model,
        train=args.train,
        data=args.data,
        generator_type=args.generator,
        predictor_model=args.predictor,
        all_samples=args.all_samples,
        cv=args.cv,
        output_path=args.out,
    )
