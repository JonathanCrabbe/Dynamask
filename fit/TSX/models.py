import os
import pickle
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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

len_of_stay = 48


class PatientData:
    """Dataset of patient vitals, demographics and lab results
    Args:
        root: Root directory of the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
        transform: Preprocessing transformation on the dataset
    """

    def __init__(
        self, root, train_ratio=0.8, shuffle=False, random_seed="1234", transform="normalize", task="mortality"
    ):
        self.data_dir = os.path.join(root, "patient_vital_preprocessed.pkl")
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)
        self.task = task
        self.pos_weight = None

        if not os.path.exists(self.data_dir):
            raise RuntimeError("Dataset not found")
        with open(self.data_dir, "rb") as f:
            self.data = pickle.load(f)

        if os.path.exists(os.path.join(root, "patient_interventions.pkl")):
            with open(os.path.join(root, "patient_interventions.pkl"), "rb") as f:
                self.intervention = pickle.load(f)

        self.n_train = int(self.train_ratio * len(self.data))
        if shuffle:
            inds = np.arange(len(self.data))
            random.shuffle(inds)
            self.data = self.data[inds]
            self.intervention = self.intervention[inds, :, :]

        if self.task == "mortality":
            X = np.array([x for (x, y, z) in self.data])
            self.train_data = X[0 : self.n_train]
            self.test_data = X[self.n_train :]
            self.train_label = np.array([y for (x, y, z) in self.data[0 : self.n_train]])
            self.test_label = np.array([y for (x, y, z) in self.data[self.n_train :]])
            self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0 : self.n_train]])
            self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train :]])

        elif self.task == "intervention":
            print("predicting intervention")
            if 0:  # suresh et al - predicts intervention state (onset, wean, stay off, stay on)
                X, y, z = self.__preprocess_predict_int__()
                choose_int = 0
                self.intervention = y[:, choose_int, :]
                self.pos_weight = np.sum(self.intervention) / np.sum(self.intervention, 0)
                self.pos_weight = 1 * self.pos_weight / self.pos_weight.sum()

            else:  # predicts interventions
                n = 3
                feat_hist = np.sum(np.sum(self.intervention, 2), 0)
                feat_idx = np.argsort(feat_hist)[::-1][:n]
                intervention_int = np.zeros((len(self.intervention), n + 1, self.intervention.shape[-1]))
                intervention_int[:, :n, :] = self.intervention[:, feat_idx, :]
                intervention_int[:, -1, :] = 1 - (np.sum(intervention_int[:, :n, :], 1) > 0).astype(int)
                self.intervention = intervention_int
                self.pos_weight = 1 / (
                    np.sum(np.sum(self.intervention, 2), 0) / (self.intervention.shape[0] * self.intervention.shape[-1])
                )
                X = np.array([x for (x, y, z) in self.data])
                z = np.array([z for (x, y, z) in self.data])

            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=88)
            for train_idx, test_idx in sss.split(X[:, :, 0], self.intervention[:, :, 0]):
                self.train_data = X[train_idx]
                self.test_data = X[test_idx]
                self.train_intervention = self.intervention[train_idx]
                self.test_intervention = self.intervention[test_idx]
                self.train_label = self.train_intervention
                self.test_label = self.test_intervention
                missing = np.array([np.mean(zz) for zz in z])
                self.train_missing = missing[train_idx]
                self.test_missing = missing[test_idx]
        self.n_train = self.train_data.shape[0]
        self.n_test = self.test_data.shape[0]
        self.feature_size = len(self.data[0][0])
        self.time = len(self.data[0][0][0])
        self.len_of_stay = self.train_data.shape[-1]
        if transform == "normalize":
            self.normalize()

    def __getitem__(self, index):
        signals, target = self.data[index]
        return signals, target

    def __len__(self):
        return len(self.data)

    def __preprocess_predict_int__(self):
        "This replicates preprocessing of suresh et al for intervention prediction"
        X_orig = np.array([x for (x, y, z) in self.data])
        y_orig = self.intervention
        z_orig = np.array([z for (x, y, z) in self.data])

        X = []
        y = []
        z = []
        T = 24
        gaptime = 4
        window = 6
        stride = 6
        for h in range(0, self.time, 6):
            if h + T + gaptime + window >= self.time - 1:
                break

            X.append(X_orig[:, :, h : h + T])
            z.append(z_orig)
            y_t1 = y_orig[:, :, range(h + T + gaptime, h + T + gaptime + window)]
            n_ints = self.intervention.shape[1]
            y_label = np.zeros((X_orig.shape[0], n_ints, 4))
            for f in range(n_ints):
                onset_patients = np.where((y_t1[:, f, 0] == 0) & (y_t1[:, f, -1] == 1))[0]
                y_label[onset_patients, f, 0] = 1
                wean_patients = np.where((y_t1[:, f, 0] == 1) & (y_t1[:, f, -1] == 0))[0]
                y_label[wean_patients, f, 1] = 1
                stay_on_patients = np.where((y_t1[:, f, 0] == 1) & (y_t1[:, f, -1] == 1))[0]
                y_label[stay_on_patients, f, 2] = 1
                stay_off_patients = np.where((y_t1[:, f, 0] == 0) & (y_t1[:, f, -1] == 0))[0]
                y_label[stay_off_patients, f, 3] = 1
            y.append(y_label)

        X = np.array(X)
        X = X.reshape((X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
        y = np.array(y)
        y = y.reshape((y.shape[0] * y.shape[1], y.shape[2], y.shape[3]))
        z = np.array(z)
        z = z.reshape((z.shape[0] * z.shape[1], -1))
        return X, y, z

    def normalize(self):  # TODO: Have multiple normalization option or possibly take in a function for the transform
        """ Calculate the mean and std of each feature from the training set
        """
        d = [x.T for x in self.train_data]
        d = np.stack(d, axis=0)
        self.feature_max = np.tile(np.max(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_min = np.tile(np.min(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_means = np.tile(np.mean(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        self.feature_std = np.tile(np.std(d.reshape(-1, self.feature_size), axis=0), (self.len_of_stay, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        self.train_data = np.array(
            [
                np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std)
                for x in self.train_data
            ]
        )
        self.test_data = np.array(
            [
                np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std)
                for x in self.test_data
            ]
        )
        # self.train_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.train_data])
        # self.test_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.test_data])


'''

class PatientData():
    """Dataset of patient vitals, demographics and lab results
    Args:
        root: Root directory of the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
        transform: Preprocessing transformation on the dataset
    """

    def __init__(self, root, train_ratio=0.8, shuffle=False, random_seed='1234', transform="normalize",task='mortality'):
        self.data_dir = os.path.join(root, 'patient_vital_preprocessed.pkl')
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)
        self.pos_weight=[1,1]

        if not os.path.exists(self.data_dir):
            raise RuntimeError('Dataset not found')
        with open(self.data_dir, 'rb') as f:
            self.data = pickle.load(f)
        if os.path.exists(os.path.join(root,'patient_interventions.pkl')):
            with open(os.path.join(root,'patient_interventions.pkl'), 'rb') as f:
                self.intervention = pickle.load(f)
            self.n_train = int(0.8*len(self.intervention))
            self.train_intervention = self.intervention[0:self.n_train,:,:]
            self.test_intervention = self.intervention[self.n_train:,:,:]
        if shuffle:
            inds = np.arange(len(self.data))
            random.shuffle(inds)
            self.data = self.data[inds]
            self.intervention = self.intervention[inds,:,:]
        self.feature_size = len(self.data[0][0])
        self.n_train = int(len(self.data) * self.train_ratio)
        self.n_test = len(self.data) - self.n_train
        self.train_data = np.array([x for (x, y, z) in self.data[0:self.n_train]])
        self.test_data = np.array([x for (x, y, z) in self.data[self.n_train:]])
        self.train_label = np.array([y for (x, y, z) in self.data[0:self.n_train]])
        self.test_label = np.array([y for (x, y, z) in self.data[self.n_train:]])
        self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0:self.n_train]])
        self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train:]])
        if transform == "normalize":
            self.normalize()

    def __getitem__(self, index):
        signals, target = self.data[index]
        return signals, target

    def __len__(self):
        return len(self.data)

    def normalize(self): # TODO: Have multiple normalization option or possibly take in a function for the transform
        """ Calculate the mean and std of each feature from the training set
        """
        d = [x.T for x in self.train_data]
        d = np.stack(d, axis=0)
        self.feature_max = np.tile(np.max(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_min = np.tile(np.min(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_means = np.tile(np.mean(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_std = np.tile(np.std(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide='ignore', invalid='ignore')
        self.train_data = np.array(
           [np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std) for
            x in self.train_data])
        self.test_data = np.array(
           [np.where(self.feature_std == 0, (x - self.feature_means), (x - self.feature_means) / self.feature_std) for
            x in self.test_data])
        # self.train_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.train_data])
        # self.test_data = np.array([ np.where(self.feature_min==self.feature_max,(x-self.feature_min),(x-self.feature_min)/(self.feature_max-self.feature_min) ) for x in self.test_data])
'''


class NormalPatientData(PatientData):
    """ Data class for the generator model that only includes patients who survived in the ICU
    """

    def __init__(self, root, train_ratio=0.8, shuffle=True, random_seed="1234", transform="normalize"):
        self.data_dir = os.path.join(root, "patient_vital_preprocessed.pkl")
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)

        if not os.path.exists(self.data_dir):
            raise RuntimeError("Dataset not found")
        with open(self.data_dir, "rb") as f:
            self.data = pickle.load(f)
        if shuffle:
            random.shuffle(self.data)
        self.feature_size = len(self.data[0][0])
        self.n_train = int(len(self.data) * self.train_ratio)
        self.n_test = len(self.data) - self.n_train

        self.train_data = np.array([x for (x, y, z) in self.data[0 : self.n_train] if y == 1])
        self.test_data = np.array([x for (x, y, z) in self.data[self.n_train :]])
        self.train_missing_samples = np.array([z for (x, y, z) in self.data[0 : self.n_train] if y == 1])
        self.test_missing_samples = np.array([z for (x, y, z) in self.data[self.n_train :]])
        self.test_label = np.array([y for (x, y, z) in self.data[self.n_train :]])
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)
        self.train_label = np.zeros((len(self.train_data), 1))
        self.train_missing = np.array([np.mean(z) for (x, y, z) in self.data[0 : self.n_train] if y == 1])
        self.test_missing = np.array([np.mean(z) for (x, y, z) in self.data[self.n_train :]])
        if transform == "normalize":
            self.normalize()


class GHGData:
    """Dataset of GHG time series
    Args:
        root: Root directory of dataset the pickled dataset
        train_ratio: train/test ratio
        shuffle: Shuffle dataset before separating train/test
    """

    def __init__(self, root, train_ratio=0.8, shuffle=True, random_seed="1234", transform=None):
        self.data_dir = os.path.join(root, "ghg_data.pkl")
        self.train_ratio = train_ratio
        self.random_seed = random.seed(random_seed)

        if not os.path.exists(self.data_dir):
            raise RuntimeError("Dataset not found")

        with open(self.data_dir, "rb") as f:
            self.data = pickle.load(f)

        print("Ignoring train ratio for this data...")

        self.feature_size = self.data["x_train"].shape[1]
        self.train_data = self.data["x_train"]
        if shuffle:
            random.shuffle(self.train_data)
        self.test_data = self.data["x_test"]
        self.train_label = self.data["y_train"]
        self.test_label = self.data["y_test"]
        self.scaler_x = self.data["scaler_x"]
        self.scaler_y = self.data["scaler_y"]
        self.train_missing = None
        self.test_missing = None
        self.n_train = len(self.train_data)
        self.n_test = len(self.test_data)

        if transform == "normalize":
            self.normalize()

    def __getitem__(self, index):
        signals, target = self.data[index]
        return signals, target

    def __len__(self):
        return len(self.data)

    def normalize(self):
        """ Calculate the mean and std of each feature from the training set
        """
        d = [x.T for x in self.train_data]
        d = np.stack(d, axis=0)
        self.feature_max = np.tile(np.max(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_min = np.tile(np.min(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_means = np.tile(np.mean(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        self.feature_std = np.tile(np.std(d.reshape(-1, self.feature_size), axis=0), (len_of_stay, 1)).T
        np.seterr(divide="ignore", invalid="ignore")
        self.train_data = np.array(
            [
                np.where(
                    self.feature_min == self.feature_max,
                    (x - self.feature_min),
                    (x - self.feature_min) / (self.feature_max - self.feature_min),
                )
                for x in self.train_data
            ]
        )
        self.test_data = np.array(
            [
                np.where(
                    self.feature_min == self.feature_max,
                    (x - self.feature_min),
                    (x - self.feature_min) / (self.feature_max - self.feature_min),
                )
                for x in self.test_data
            ]
        )


class ConvClassifier(nn.Module):
    def __init__(
        self,
        feature_size,
        n_state,
        hidden_size,
        regres=True,
        return_all=False,
        seed=random.seed("2019"),
        data="simulation",
    ):
        super(ConvClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.regres = regres
        self.return_all = return_all
        self.data = data

        # Input to torch Conv
        self.regressor = nn.Sequential(
            torch.nn.Conv1d(in_channels=feature_size, out_channels=self.hidden_size, kernel_size=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv1d(in_channels=self.hidden_size, out_channels=self.n_state, kernel_size=1, padding=0),
        )

    def forward(self, input, **kwargs):
        return self.regressor(input)[:, :, -1]


class StateClassifierMIMIC(nn.Module):
    def __init__(
        self,
        feature_size,
        n_state,
        hidden_size,
        rnn="LSTM",
        regres=True,
        bidirectional=False,
        return_all=False,
        seed=random.seed("2019"),
        data="simulation",
    ):
        super(StateClassifierMIMIC, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.data = data
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn1 = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
            self.rnn2 = nn.GRU(self.hidden_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn1 = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
            self.rnn2 = nn.LSTM(self.hidden_size, self.hidden_size, bidirectional=bidirectional).to(self.device)

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            # nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(self.hidden_size, self.n_state),
        )
        # nn.Softmax(-1))

    def forward(self, input, past_state=None, **kwargs):
        input = input.permute(2, 0, 1).to(self.device)
        self.rnn1.to(self.device)
        self.rnn2.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            # past_state1 = torch.normal(mean=0,std=1, size=[1, input.shape[1], self.hidden_size]).to(self.device)
            # past_state2 = torch.normal(mean=0,std=1, size=[1, input.shape[1], self.hidden_size]).to(self.device)
            past_state1 = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
            past_state2 = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn1(input, past_state1)
            all_encodings, encoding = self.rnn2(all_encodings, past_state2)
        else:
            all_encodings, (encoding, state) = self.rnn1(input, (past_state1, past_state1))
            all_encodings, (encoding, state) = self.rnn2(all_encodings, (past_state2, past_state2))
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1] * all_encodings.shape[0], -1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1))
        else:
            return encoding.view(encoding.shape[1], -1)


class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size,
        n_state,
        hidden_size,
        rnn="GRU",
        regres=True,
        bidirectional=False,
        return_all=False,
        seed=random.seed("2019"),
        data="simulation",
    ):
        super(StateClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.data = data
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.hidden_size, self.n_state),
        )
        # nn.Softmax(-1))

    def forward(self, input, past_state=None, **kwargs):
        input = input.permute(2, 0, 1).to(self.device)
        self.rnn.to(self.device)
        self.regressor.to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        if self.regres:
            if not self.return_all:
                return self.regressor(encoding.view(encoding.shape[1], -1))
            else:
                reshaped_encodings = all_encodings.view(all_encodings.shape[1] * all_encodings.shape[0], -1)
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1))
        else:
            return encoding.view(encoding.shape[1], -1)


class EncoderRNN(nn.Module):
    def __init__(
        self,
        feature_size,
        hidden_size,
        rnn="GRU",
        regres=True,
        bidirectional=False,
        return_all=False,
        seed=random.seed("2019"),
        data="mimic",
        n_state=1,
        return_multi=False,
    ):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rnn_type = rnn
        self.regres = regres
        self.return_all = return_all
        self.return_multi = False
        # Input to torch LSTM should be of size (seq_len, batch, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)
        else:
            self.rnn = nn.LSTM(feature_size, self.hidden_size, bidirectional=bidirectional).to(self.device)

        if data == "mimic":
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(num_features=self.hidden_size), nn.Dropout(0.5), nn.Linear(self.hidden_size, n_state)
            )  # ,
            # nn.Sigmoid())
        elif data == "ghg":
            self.regressor = nn.Sequential(  # nn.BatchNorm1d(self.hidden_size),
                nn.Linear(self.hidden_size, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 200),
                nn.LeakyReLU(),
                # nn.Dropout(0.5),
                nn.Linear(200, n_state),
            )
        elif "simulation" in data:
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(num_features=self.hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.hidden_size, n_state),
            )  # ,
            # nn.Sigmoid())

    def forward(self, input, past_state=None):
        input = input.permute(2, 0, 1).to(self.device)
        if not past_state:
            #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
            past_state = torch.zeros([1, input.shape[1], self.hidden_size]).to(self.device)
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(input, past_state)
        else:
            all_encodings, (encoding, state) = self.rnn(input, (past_state, past_state))
        if self.regres:
            if not self.return_all:
                if not self.return_multi:
                    return self.regressor(encoding.view(encoding.shape[1], -1))
                else:
                    multiclass = torch.cuda.FloatTensor(encoding.shape[1], 2).fill_(0)
                    multiclass[:, 1] = torch.sigmoid(self.regressor(encoding.view(encoding.shape[1], -1))[:, 0])
                    multiclass[:, 0] = 1 - multiclass[:, 1]
                    return multiclass
            else:
                # print('before: ', all_encodings[-1,-1,:])
                reshaped_encodings = all_encodings.view(all_encodings.shape[1] * all_encodings.shape[0], -1)
                # print('after: ', reshaped_encodings[-1:,:].data.cpu().numpy())
                return torch.t(self.regressor(reshaped_encodings).view(all_encodings.shape[0], -1))
        else:
            return encoding.view(encoding.shape[1], -1)


class RnnVAE(nn.Module):
    def __init__(self, feature_size, hidden_size, bidirectional=False, seed=random.seed("2019")):
        super(RnnVAE, self).__init__()
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.seed = seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # q(Zt|X0:t)
        self.encoder = nn.GRU(self.feature_size, 2 * self.hidden_size, bidirectional=bidirectional)
        # P(Xt|Zt)
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, 400), nn.ReLU(), nn.Linear(400, self.feature_size), nn.Sigmoid()
        )

    def encode(self, input):
        input = input.permute(2, 0, 1)
        #  Size of hidden states: (num_layers * num_directions, batch, hidden_size)
        past_state = torch.zeros([1, input.shape[1], self.hidden_size * 2]).to(self.device)
        _, encoding = self.encoder(input, past_state)
        mu = nn.ReLU()(encoding[:, :, : self.hidden_size]).view(-1, self.hidden_size)
        logvar = nn.ReLU()(encoding[:, :, self.hidden_size :]).view(-1, self.hidden_size)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        return z


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device):
        super(DecoderRNN, self).__init__()
        self.output_size = output_size
        self.encoding_size = hidden_size
        self.rnn = nn.GRUCell(output_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.device = device

    def forward(self, encoding, out_len, past_state=None):
        output = torch.zeros([out_len, encoding.shape[0], self.output_size])
        if not past_state:
            past_state = torch.zeros(encoding.shape).to(self.device)
        for i in range(out_len, 0, -1):
            print(encoding.shape, past_state.shape)
            encoding = self.rnn(encoding, past_state)
            past_state = nn.Softmax()(self.out(past_state))
            output[i - 1, :, :] = past_state
        return output


class LR(nn.Module):
    def __init__(self, feature_size):
        super(LR, self).__init__()
        self.feature_size = feature_size
        self.net = nn.Sequential(nn.Linear(self.feature_size, 1), nn.Sigmoid())
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, x):
        x = x.to(self.device)
        x = x.mean(dim=2).reshape((x.shape[0], -1))
        if len(x.shape) == 3:
            x = x.view(-1, self.feature_size)
        risk = self.net(x)
        return risk


class RiskPredictor(nn.Module):
    def __init__(self, encoding_size):
        super(RiskPredictor, self).__init__()
        self.encoding_size = encoding_size
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.net = nn.Sequential(nn.Linear(self.encoding_size, 500), nn.ReLU(True), nn.Dropout(0.5), nn.Linear(500, 1))

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def forward(self, x):
        logits = self.net(x)
        risk = self.temperature_scale(logits)
        # risk = nn.Sigmoid()(self.net(x))
        return risk

    def forward_logit(self, x):
        logits = self.net(x)
        return logits


class AttentionModel(torch.nn.Module):
    def __init__(self, hidden_size, feature_size, data):
        super(AttentionModel, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        # self.W_s1 = nn.Linear(hidden_size, 350)
        # self.W_s2 = nn.Linear(350, 30)
        self.W_s1 = nn.Linear(hidden_size, 1)
        # self.fc_layer = nn.Linear(30 * hidden_size, 2000)
        self.rnn = nn.GRU(self.feature_size, hidden_size)
        if data == "mimic":
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(num_features=hidden_size), nn.Dropout(0.5), nn.Linear(hidden_size, 1), nn.Sigmoid()
            )
        elif data == "ghg":
            self.regressor = nn.Sequential(  # nn.BatchNorm1d(self.hidden_size),
                nn.Linear(hidden_size, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 200),
                nn.LeakyReLU(),
                nn.Linear(200, 200),
                nn.LeakyReLU(),
                # nn.Dropout(0.5),
                nn.Linear(200, 1),
            )
        elif "simulation" in data:
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(num_features=hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid(),
            )

    def attention_net(self, lstm_output):
        attn_weight_vector = F.tanh(self.W_s1(lstm_output))
        attn_weight_vector = torch.nn.Softmax(dim=1)(attn_weight_vector)
        scaled_latent = lstm_output * attn_weight_vector
        return torch.sum(scaled_latent, dim=1), attn_weight_vector

    def forward(self, input):
        input = input.to(self.device)
        batch_size = input.shape[0]
        input = input.permute(2, 0, 1)  # Input to GRU should be (seq_len, batch, input_size)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(
            self.device
        )  # (num_layers * num_directions, batch, hidden_size)

        output, final_hidden_state = self.rnn(input, h_0)  # output.size() =  (seq_len, batch, hidden_size)
        # final_hidden_state.size() = (1, batch, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch, seq_len, hidden_size)

        concept_vector, attn_weights = self.attention_net(output)  # attn_weight_matrix.size() = (batch_size, num_seq)
        # hidden_matrix = torch.bmm(attn_weight_matrix, output)   # hidden_matrix.size() = (batch_size, r, hidden_size)
        # fc_out = self.fc_layer(hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2]))
        p = self.regressor(concept_vector)
        return p

    def get_attention_weights(self, input):
        input = input.to(self.device)
        batch_size = input.shape[0]
        input = input.permute(2, 0, 1)  # Input to GRU should be (seq_len, batch, input_size)
        h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size)).to(
            self.device
        )  # (num_layers * num_directions, batch, hidden_size)

        output, final_hidden_state = self.rnn(input, h_0)  # output.size() =  (seq_len, batch, hidden_size)
        # final_hidden_state.size() = (1, batch, hidden_size)
        output = output.permute(1, 0, 2)  # output.size() = (batch, seq_len, hidden_size)

        _, attn_weights = self.attention_net(output)
        return attn_weights


class RETAIN(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_emb=128,
        dropout_input=0.8,
        dropout_emb=0.5,
        dim_alpha=128,
        dim_beta=128,
        dropout_context=0.5,
        dim_output=2,
        l2=0.0001,
        batch_first=True,
    ):
        super(RETAIN, self).__init__()
        self.batch_first = batch_first
        self.embedding = nn.Sequential(
            nn.Dropout(p=dropout_input), nn.Linear(dim_input, dim_emb, bias=False), nn.Dropout(p=dropout_emb)
        )
        nn.init.xavier_normal(self.embedding[1].weight)

        self.rnn_alpha = nn.GRU(input_size=dim_emb, hidden_size=dim_alpha, num_layers=1, batch_first=self.batch_first)

        self.alpha_fc = nn.Linear(in_features=dim_alpha, out_features=1)
        nn.init.xavier_normal(self.alpha_fc.weight)
        self.alpha_fc.bias.data.zero_()

        self.rnn_beta = nn.GRU(input_size=dim_emb, hidden_size=dim_beta, num_layers=1, batch_first=self.batch_first)

        self.beta_fc = nn.Linear(in_features=dim_beta, out_features=dim_emb)
        nn.init.xavier_normal(self.beta_fc.weight, gain=nn.init.calculate_gain("tanh"))
        self.beta_fc.bias.data.zero_()

        self.output = nn.Sequential(
            nn.Dropout(p=dropout_context), nn.Linear(in_features=dim_emb, out_features=dim_output)
        )
        nn.init.xavier_normal(self.output[1].weight)
        self.output[1].bias.data.zero_()

    def forward(self, x, lengths):
        if self.batch_first:
            batch_size, max_len = x.size()[:2]
        else:
            max_len, batch_size = x.size()[:2]
        # emb -> batch_size X max_len X dim_emb
        emb = self.embedding(x)
        # print('Embedding: ', emb.shape)

        packed_input = pack_padded_sequence(emb, lengths, batch_first=self.batch_first)
        g, _ = self.rnn_alpha(packed_input)

        # alpha_unpacked -> batch_size X max_len X dim_alpha
        alpha_unpacked, _ = pad_packed_sequence(g, batch_first=self.batch_first)
        # print(alpha_unpacked.shape)

        # mask -> batch_size X max_len X 1
        mask = Variable(
            torch.FloatTensor(
                [[1.0 if i < lengths[idx] else 0.0 for i in range(max_len)] for idx in range(batch_size)]
            ).unsqueeze(2),
            requires_grad=False,
        )
        # print('Mask: ', mask.shape)
        if next(self.parameters()).is_cuda:  # returns a boolean
            mask = mask.cuda()

        # e => batch_size X max_len X 1
        e = self.alpha_fc(alpha_unpacked)

        def masked_softmax(batch_tensor, mask):
            exp = torch.exp(batch_tensor)
            masked_exp = exp * mask
            sum_masked_exp = torch.sum(masked_exp, dim=1, keepdim=True)
            return masked_exp / sum_masked_exp

        # Alpha = batch_size X max_len X 1
        # alpha value for padded visits (zero) will be zero
        alpha = masked_softmax(e, mask)

        h, _ = self.rnn_beta(packed_input)

        # beta_unpacked -> batch_size X max_len X dim_beta
        beta_unpacked, _ = pad_packed_sequence(h, batch_first=self.batch_first)

        # Beta -> batch_size X max_len X dim_emb
        # beta for padded visits will be zero-vectors
        beta = torch.nn.functional.tanh(self.beta_fc(beta_unpacked) * mask)

        # context -> batch_size X (1) X dim_emb (squeezed)
        # Context up to i-th visit context_i = sum(alpha_j * beta_j * emb_j)
        # Vectorized sum
        context = torch.bmm(torch.transpose(alpha, 1, 2), beta * emb).squeeze(1)

        # without applying non-linearity
        logit = self.output(context)

        return logit, alpha, beta
