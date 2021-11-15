import argparse
import os
import pickle

import numpy as np

SIG_NUM = 3
STATE_NUM = 1
P_S0 = [0.5]

correlated_feature = [0, 0]  # Features that re correlated with the important feature in each state

imp_feature = [1, 2]  # Feature that is always set as important
scale = [[0.1, 1.6, 0.5], [-0.1, -0.4, -1.5]]  # Scaling factor for distribution mean in each state
trans_mat = np.array([[0.1, 0.9], [0.1, 0.9]])
# print(trans_mat.shape)


def init_distribution_params():
    # Covariance matrix is constant across states but distribution means change based on the state value
    state_count = np.power(2, STATE_NUM)
    # corr = abs(np.random.randn(SIG_NUM))
    cov = np.eye(SIG_NUM) * 0.8
    covariance = []
    for i in range(state_count):
        c = cov.copy()
        c[imp_feature[i], correlated_feature[i]] = 0.01
        c[correlated_feature[i], imp_feature[i]] = 0.01
        c = c + np.eye(SIG_NUM) * 1e-3
        # print(c)
        covariance.append(c)
    covariance = np.array(covariance)
    mean = []
    for i in range(state_count):
        # m = (np.random.randn(SIG_NUM))*scale[i]
        m = scale[i]
        mean.append(m)
        # print(m)
    mean = np.array(mean)
    return mean, covariance


def next_state(previous_state, t):
    # params = [(abs(p-0.1)+timing_factor)/2. for p in previous_state]
    # print(params,previous_state)
    # params = [abs(p - 0.1) for p in previous_state]
    # print(previous_state)
    # params = [abs(p) for p in trans_mat[int(previous_state),1-int(previous_state)]]
    # params = trans_mat[int(previous_state),1-int(previous_state)]
    if previous_state == 1:
        params = 0.95
    else:
        params = 0.05
    # params = 0.2
    # print('previous', previous_state)
    params = params - float(t / 500) if params > 0.8 else params
    # print('transition probability',params)
    next = np.random.binomial(1, params)
    return next


# def state_decoder(state_one_hot):
#    base = 1
#    state = 0
#    for digit in state_one_hot:
#        state = state + base*digit
#        base = base * 2
#    return state


def state_decoder(previous, next):
    return int(next * (1 - previous) + (1 - next) * (previous))


def create_signal(sig_len, mean, cov):
    signal = []
    states = []
    y = []
    importance = []
    y_logits = []

    previous = np.random.binomial(1, P_S0)[0]
    delta_state = 0
    state_n = None
    for i in range(sig_len):
        # next = next_state(previous, i)

        next = next_state(previous, delta_state)
        # state_n = state_decoder(previous,next)
        state_n = next

        if state_n == previous:
            delta_state += 1
        else:
            delta_state = 0

        # if state_n!=previous:
        imp_sig = np.zeros(3)
        if state_n != previous or i == 0:
            imp_sig[imp_feature[state_n]] = 1
        # imp_sig[correlated_feature[state_n]] = 1
        # else:
        #    imp_sig = [0, 0, 0]

        importance.append(imp_sig)
        sample = np.random.multivariate_normal(mean[state_n], cov[state_n])
        previous = state_n
        signal.append(sample)
        y_logit = logit(sample[imp_feature[state_n]])
        y_label = np.random.binomial(1, y_logit)

        # print('previous state:',previous,'next state probability:', next, 'delta_state:',delta_state,'current state:',state_n, 'mean:', mean[state_n], 'cov:', cov[state_n],'ylogit', y_logit)

        y.append(y_label)
        y_logits.append(y_logit)
        states.append(state_n)
    signal = np.array(signal)
    y = np.array(y)
    importance = np.array(importance)
    # print(importance.shape)
    return signal.T, y, states, importance, y_logits


def decay(x):
    return [0.9 * (1 - 0.1) ** x, 0.9 * (1 - 0.1) ** x]


def logit(x):
    return 1.0 / (1 + np.exp(-2 * (x)))


def normalize(train_data, test_data, config="mean_normalized"):
    """ Calculate the mean and std of each feature from the training set
    """
    feature_size = train_data.shape[1]
    len_of_stay = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)
    if config == "mean_normalized":
        feature_means = np.tile(np.mean(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_std = np.tile(np.std(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        train_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for x in train_data]
        )
        test_data_n = np.array(
            [np.where(feature_std == 0, (x - feature_means), (x - feature_means) / feature_std) for x in test_data]
        )
    elif config == "zero_to_one":
        feature_max = np.tile(np.max(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        feature_min = np.tile(np.min(d.reshape(-1, feature_size), axis=0), (len_of_stay, 1)).T
        train_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in train_data])
        test_data_n = np.array([(x - feature_min) / (feature_max - feature_min) for x in test_data])
    return train_data_n, test_data_n


def create_dataset(count, signal_len):
    dataset = []
    labels = []
    importance_score = []
    states = []
    label_logits = []
    mean, cov = init_distribution_params()
    for num in range(count):
        sig, y, state, importance, y_logits = create_signal(signal_len, mean, cov)
        dataset.append(sig)
        labels.append(y)
        importance_score.append(importance.T)
        states.append(state)
        label_logits.append(y_logits)
    dataset = np.array(dataset)
    labels = np.array(labels)
    importance_score = np.array(importance_score)
    states = np.array(states)
    label_logits = np.array(label_logits)
    n_train = int(len(dataset) * 0.8)
    train_data = dataset[:n_train]
    test_data = dataset[n_train:]
    # train_data_n, test_data_n = normalize(train_data, test_data)
    train_data_n = train_data
    test_data_n = test_data
    save_dir = "./data/state/"
    if not os.path.exists(save_dir):
        print(f"Creating saving directory {save_dir}.")
        os.makedirs(save_dir)
    print(f"Saving data in {save_dir}.")
    with open(os.path.join(save_dir, "state_dataset_x_train.pkl"), "wb") as f:
        pickle.dump(train_data_n, f)
    with open(os.path.join(save_dir, "state_dataset_x_test.pkl"), "wb") as f:
        pickle.dump(test_data_n, f)
    with open(os.path.join(save_dir, "state_dataset_y_train.pkl"), "wb") as f:
        pickle.dump(labels[:n_train], f)
    with open(os.path.join(save_dir, "state_dataset_y_test.pkl"), "wb") as f:
        pickle.dump(labels[n_train:], f)
    with open(os.path.join(save_dir, "state_dataset_importance_train.pkl"), "wb") as f:
        pickle.dump(importance_score[:n_train], f)
    with open(os.path.join(save_dir, "state_dataset_importance_test.pkl"), "wb") as f:
        pickle.dump(importance_score[n_train:], f)
    with open(os.path.join(save_dir, "state_dataset_logits_train.pkl"), "wb") as f:
        pickle.dump(label_logits[:n_train], f)
    with open(os.path.join(save_dir, "state_dataset_logits_test.pkl"), "wb") as f:
        pickle.dump(label_logits[n_train:], f)
    with open(os.path.join(save_dir, "state_dataset_states_train.pkl"), "wb") as f:
        pickle.dump(states[:n_train], f)
    with open(os.path.join(save_dir, "state_dataset_states_test.pkl"), "wb") as f:
        pickle.dump(states[n_train:], f)

    return dataset, labels, states


if __name__ == "__main__":
    if not os.path.exists("./data"):
        os.mkdir("./data")
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_len", type=int, default=100, help="Length of the signal to generate")
    parser.add_argument("--signal_num", type=int, default=1000, help="Number of the signals to generate")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    np.random.seed(234)
    dataset, labels, states = create_dataset(args.signal_num, args.signal_len)

    if args.plot:
        import matplotlib.pyplot as plt

        f, (x1, x2) = plt.subplots(2, 1)
        for id in range(len(labels)):
            for i, sample in enumerate(dataset[id]):
                if labels[id, i]:
                    x1.scatter(sample[0], sample[1], c="r")
                else:
                    x1.scatter(sample[0], sample[1], c="b")
                if states[id, i]:
                    x2.scatter(sample[0], sample[1], c="b")
                else:
                    x2.scatter(sample[0], sample[1], c="r")
            x1.set_title("Distribution based on label")
            x2.set_title("Distribution based on state")
        plt.show()
