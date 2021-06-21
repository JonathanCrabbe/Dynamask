import torch
import os, glob
from abc import ABC, abstractmethod
from TSX.utils import train_model, train_model_rt, train_model_rt_rg, plot_importance, logistic, test_model_rt, test, replace_and_predict
from TSX.models import EncoderRNN, LR, AttentionModel
from TSX.generator import FeatureGenerator, train_joint_feature_generator, train_feature_generator, CarryForwardGenerator, DLMGenerator, JointFeatureGenerator
import seaborn as sns
sns.set()
import matplotlib.colors as mcolors
import matplotlib
import numpy as np
import pickle as pkl
import json
import time

import lime
import lime.lime_tabular

font={'family': 'normal','weight': 'bold','size':82}
matplotlib.rc('font',**font)

#mimic plot configs
xkcd_colors = mcolors.XKCD_COLORS
color_map = [list(xkcd_colors.keys())[k] for k in
             np.random.choice(range(len(xkcd_colors)), 28, replace=False)]
color_map = ['#00998F', '#C20088', '#0075DC', '#E0FF66', '#4C005C', '#191919', '#FF0010', '#2BCE48', '#FFCC99', '#808080',
             '#740AFF', '#8F7C00', '#9DCC00', '#F0A3FF', '#94FFB5', '#FFA405', '#FFA8BB', '#426600', '#005C31', '#5EF1F2',
             '#993F00', '#990000', '#003380', '#990000', '#FFFF80', '#FF5005', '#FFFF00','#FF0010', '#FFCC99']

intervention_list = ['vent', 'vaso', 'adenosine', 'dobutamine', 'dopamine', 'epinephrine', 'isuprel', 'milrinone', 'norepinephrine', 'phenylephrine', 'vasopressin', 'colloid_bolus', 'crystalloid_bolus', 'nivdurations']

feature_map_mimic = ['ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 'HEMOGLOBIN',
           'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 'PLATELET', 'POTASSIUM', 'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'HeartRate' ,
           'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose','Temp']

#simulation plot configs
feature_map_simulation = ['feature 0', 'feature 1', 'feature 2']

feature_map = {'mimic':feature_map_mimic, 'simulation': feature_map_simulation,'simulation_spike':feature_map_simulation}

simulation_color_map = ['#e6194B', '#469990', '#000000','#4363d8', '#f58231', '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabebe',  '#e6beff', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#3cb44b','#ffe119']


#ghg plot configs
feature_map_ghg = [str(i+1) for i in range(15)]
scatter_map_ghg={}
for i in range(15):
    scatter_map_ghg[feature_map_ghg[i]]=[]
width=500
height=560
scatter_map_ghg['1'] = [174,height-60]
scatter_map_ghg['2'] = [59,height-81]
scatter_map_ghg['3'] = [126,height-100]
scatter_map_ghg['4'] = [181,height-161]
scatter_map_ghg['5'] = [101,height-200]
scatter_map_ghg['6'] = [294,height-289]
scatter_map_ghg['7'] = [106,height-226]
scatter_map_ghg['8'] = [178,height-291]
scatter_map_ghg['9'] = [141,height-315]
scatter_map_ghg['10'] = [405,height-420]
scatter_map_ghg['11'] = [190,height-388]
scatter_map_ghg['12'] = [291,height-453]
scatter_map_ghg['13'] = [385,height-489]
scatter_map_ghg['14'] = [383,height-516]
scatter_map_ghg['15'] = [310,height-212]


class Experiment(ABC):
    def __init__(self, train_loader, valid_loader, test_loader, data='mimic'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.data = data
        if not os.path.exists('./ckpt'):
            os.mkdir('./ckpt')
        self.ckpt_path ='./ckpt/' + self.data
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)


    @abstractmethod
    def run(self):
        raise RuntimeError('Function not implemented')

    def train(self, n_epochs, learn_rt=False):
        train_start_time = time.time()
        if self.data=='mimic':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)
        
        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment, data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if self.data == 'mimic' or 'simulation' in self.data:
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)
            elif self.data == 'ghg':
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device, self.experiment,data=self.data)
        print("Generator training time = ", time.time()-train_start_time)


class Baseline(Experiment):
    """ Baseline mortality prediction using a logistic regressions model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, experiment='baseline',data='mimic'):
        super(Baseline, self).__init__(train_loader, valid_loader, test_loader)
        self.model = LR(feature_size).to(self.device)
        self.experiment = experiment
        self.data=data

    def run(self, train):
        if train:
            self.train(n_epochs=250)
        else:
            if os.path.exists(self.ckpt_path + '/' +  str(self.experiment) + '.pt'):
                self.model.load_state_dict(torch.load('./ckpt/'+ self.data + '/' + str(self.experiment) + '.pt'))
                _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                print('Loading model with AUC: ', auc_test)
            else:
                raise RuntimeError('No saved checkpoint for this model')


class EncoderPredictor(Experiment):
    """ Baseline mortality prediction using an encoder to encode patient status, and a risk predictor to predict risk of mortality
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, encoding_size, rnn_type='GRU', experiment='risk_predictor',simulation=False,data='mimic', model='RNN'):
        super(EncoderPredictor, self).__init__(train_loader, valid_loader, test_loader, data=data)
        if model=='RNN':
            self.model = EncoderRNN(feature_size, encoding_size, rnn=rnn_type, regres=True, return_all=False,data=data)
        elif model=='LR':
            self.model = LR(feature_size)
        elif model=='attention':
            self.model = AttentionModel(encoding_size, feature_size, data)
        self.model_type = model
        self.experiment = experiment
        self.data = data

    def run(self, train,n_epochs, **kwargs):
        if train:
            self.train(n_epochs=n_epochs, learn_rt=self.data!='mimic')
        else:
            path = './ckpt/' + self.data + '/' + str(self.experiment) + '_' + self.model_type + '.pt'
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path))

                if self.data=='mimic':
                    _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
                else:#ghg is regression
                    test_loss, _, _, auc_test, _ = test_model_rt(self.model, self.test_loader)
            else:
                raise RuntimeError('No saved checkpoint for this model')

    def train(self, n_epochs, learn_rt=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-3)
        if not learn_rt:
            train_model(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment+'_'+self.model_type, data=self.data)
            # Evaluate performance on held-out test set
            _, _, auc_test, correct_label, test_loss = test(self.test_loader, self.model, self.device)
            print('\nFinal performance on held out test set ===> AUC: ', auc_test)
        else:
            if 'simulation' in self.data:
                print('training rt')
                train_model_rt(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                            self.experiment+'_'+self.model_type, data=self.data)
                # Evaluate performance on held-out test set
                _, _, _, auc, correct_label = test_model_rt(self.model, self.valid_loader)
                print('\nFinal performance on held out test set ===> AUC: ', auc)

            else:
                #only for ghg data
                train_model_rt_rg(self.model, self.train_loader, self.valid_loader, optimizer, n_epochs, self.device,
                                               self.experiment,data=self.data)


class BaselineExplainer(Experiment):
    """ Baseline explainability methods
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, data_class, experiment='baseline_explainer', data='mimic', baseline_method='lime',**kwargs):
        super(BaselineExplainer, self).__init__(train_loader, valid_loader, test_loader, data=data)
        self.experiment = experiment
        self.data_class = data_class
        self.baseline_method = baseline_method
        self.input_size = feature_size
        self.learned_risk = True
        if data == 'mimic':
            self.timeseries_feature_size = len(feature_map_mimic)
        else:
            self.timeseries_feature_size = feature_size

        # Build the risk predictor and load checkpoint
        with open('config.json') as config_file:
            configs = json.load(config_file)[data]['risk_predictor']
        if 'simulation' in self.data:
            if not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            else:
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
            self.feature_map = feature_map_simulation
        else:
            if self.data == 'mimic':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
                self.feature_map = feature_map_mimic
                self.risk_predictor = self.risk_predictor.to(self.device)
            elif self.data == 'ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=configs['encoding_size'],
                                                 rnn=configs['rnn_type'], regres=True, data=data)
                self.feature_map = feature_map_ghg
                self.risk_predictor = self.risk_predictor.to(self.device)
        self.risk_predictor.to(self.device)
        self.risk_predictor.eval()

    def predictor_wrapper(self, sample):
        """
        In order to use the lime explainer library we need to go back and forth between numpy library (compatible with Lime)
        and torch (Compatible with the predictor model). This wrapper helps with this
        :param sample: input sample for the predictor (type: numpy array)
        :return: one-hot model output (type: numpy array)
        """
        torch_in = torch.Tensor(sample).reshape(len(sample),-1,1)
        torch_in.to(self.device)
        out = self.risk_predictor(torch_in)
        one_hot_out = np.concatenate((out.detach().cpu().numpy(), out.detach().cpu().numpy()), axis=1)
        one_hot_out[:,1] = 1-one_hot_out[:,0]
        return one_hot_out

    def run(self, train, n_epochs, samples_to_analyze):
        self.train(n_epochs=n_epochs, learn_rt=self.data=='ghg')
        testset = list(self.test_loader.dataset)
        test_signals = torch.stack(([x[0] for x in testset])).to(self.device)
        matrix_test_dataset = test_signals.cpu().numpy()
        explanation = []
        exec_time = []
        for test_sample_ind, test_sample in enumerate(samples_to_analyze):
            exp_sample=[]
            lime_start = time.time()
            for tttt in range(matrix_test_dataset.shape[2]):
                exp = self.explainer.explain_instance(np.mean(matrix_test_dataset[test_sample,:,:tttt+1],axis=1), self.predictor_wrapper, labels=['feature '+ str(i) for i in range(len(self.feature_map))], top_labels=2)
                exp_sample.append(exp.as_list())
            explanation.append(exp_sample)
            exec_time.append(time.time()-lime_start)
        exec_time = np.array(exec_time)
        # print("Execution time of lime for subject %d= %.3f +/- %.3f"%(test_sample, np.mean(exec_time), np.std(np.array(exec_time))))
        return explanation

    def train(self, n_epochs, learn_rt=False):
        trainset = list(self.train_loader.dataset)
        signals = torch.stack(([x[0] for x in trainset])).to(self.device)
        matrix_train_dataset = signals.mean(dim=2).cpu().numpy()
        if self.baseline_method == 'lime':
            self.explainer = lime.lime_tabular.LimeTabularExplainer(matrix_train_dataset, feature_names=self.feature_map+['gender', 'age', 'ethnicity', 'first_icu_stay'], discretize_continuous=True)


class FeatureGeneratorExplainer(Experiment):
    """ Experiment for generating feature importance over time using a generative model
    """
    def __init__(self,  train_loader, valid_loader, test_loader, feature_size, patient_data, output_path, generator_hidden_size=80, prediction_size=1, generator_type='RNN_generator', predictor_model='RNN', experiment='feature_generator_explainer', data='mimic', conditional=True,**kwargs):
        super(FeatureGeneratorExplainer, self).__init__(train_loader, valid_loader, test_loader, data)
        self.generator_type = generator_type
        if self.generator_type == 'RNN_generator':
            self.generator = FeatureGenerator(feature_size, hidden_size=generator_hidden_size, prediction_size=prediction_size,data=data,conditional=conditional).to(self.device)
            self.conditional=conditional
        elif self.generator_type == 'carry_forward_generator':
            self.generator = CarryForwardGenerator(feature_size).to(self.device)
        elif 'joint_RNN_generator' in self.generator_type :
            self.generator = JointFeatureGenerator(feature_size, data=data).to(self.device) # TODO setup the right encoding size
        elif self.generator_type == 'dlm_joint_generator':
            self.generator = DLMGenerator(feature_size).to(self.device) # TODO setup the right encoding size
        else:
            raise RuntimeError('Undefined generator!')

        if data == 'mimic':
            self.timeseries_feature_size = len(feature_map_mimic)
        else:
            self.timeseries_feature_size = feature_size

        self.feature_size = feature_size
        self.input_size = feature_size
        self.patient_data = patient_data
        self.experiment = experiment
        self.predictor_model = predictor_model
        self.prediction_size = prediction_size
        self.generator_hidden_size = generator_hidden_size
        self.output_path = os.path.join(output_path, data)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
        if self.generator_type!='RNN_generator':
            self.conditional=None

        #this is used to see the difference between true risk vs learned risk for simulations
        self.learned_risk = True
        trainset = list(self.train_loader.dataset)
        self.feature_dist = torch.stack([x[0] for x in trainset])
        if self.data == 'mimic':
            self.feature_dist_0 = torch.stack([x[0] for x in trainset if x[1]==0])
            self.feature_dist_1 = torch.stack([x[0] for x in trainset if x[1]==1])
        else:
            self.feature_dist_0=self.feature_dist
            self.feature_dist_1=self.feature_dist

        # TODO: instead of hard coding read from json
        if 'simulation' in self.data:
            if not self.learned_risk:
                self.risk_predictor = lambda signal,t:logistic(2.5*(signal[0, t] * signal[0, t] + signal[1,t] * signal[1,t] + signal[2, t] * signal[2, t] - 1))
            else:
                if self.data=='simulation_spike':
                    if self.predictor_model == 'RNN':
                        self.risk_predictor = EncoderRNN(feature_size,hidden_size=50,rnn='GRU',regres=True, return_all=False,data=data)
                    elif self.predictor_model == 'attention':
                        self.risk_predictor = AttentionModel(feature_size=feature_size, hidden_size=50, data=data)
                    self.risk_predictor_attention = AttentionModel(feature_size=feature_size, hidden_size=50, data=data)
                else:
                    if self.predictor_model == 'RNN':
                        self.risk_predictor = EncoderRNN(feature_size,hidden_size=100,rnn='GRU',regres=True, return_all=False,data=data)
                    elif self.predictor_model == 'attention':
                        self.risk_predictor = AttentionModel(feature_size=feature_size, hidden_size=100, data=data)
                    self.risk_predictor_attention = AttentionModel(feature_size=feature_size, hidden_size=100, data=data)
            self.feature_map = feature_map_simulation
            self.risk_predictor = self.risk_predictor.to(self.device)
            self.risk_predictor_attention = self.risk_predictor_attention.to(self.device)
        else:
            if self.data=='mimic':
                if self.predictor_model == 'RNN':
                    self.risk_predictor = EncoderRNN(self.input_size, hidden_size=150, rnn='GRU', regres=True,data=data)
                elif self.predictor_model == 'attention':
                    self.risk_predictor = AttentionModel(feature_size=feature_size, hidden_size=150, data=data)
                self.risk_predictor_attention = AttentionModel(feature_size=feature_size, hidden_size=150, data=data)
                self.feature_map = feature_map_mimic
            elif self.data=='ghg':
                self.risk_predictor = EncoderRNN(self.input_size, hidden_size=500, rnn='LSTM', regres=True,data=data)
                self.feature_map = feature_map_ghg
            self.risk_predictor = self.risk_predictor.to(self.device)
            self.risk_predictor_attention = self.risk_predictor_attention.to(self.device)

    def select_top_features(self, samples_to_analyze, sub_features, alpha=0.01):
        check_path = glob.glob(os.path.join(self.ckpt_path,'*_generator.pt'))[0]
        selected_subgroups = {}
        if not os.path.exists(check_path):
            raise RuntimeError('No saved checkpoint for this model')

        if not 'simulation' in self.data:
            self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path,'risk_predictor_%s.pt'%self.predictor_model)))
            self.risk_predictor.to(self.device)
            self.risk_predictor.eval()
            _, _, auc, correct_label, _ = test(self.test_loader, self.risk_predictor, self.device)
        else: #simulated data
            if self.learned_risk:
                self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor_%s.pt'%self.predictor_model)))
                self.risk_predictor.to(self.device)
                self.risk_predictor.eval()
                _, _, _, auc,  _ = test_model_rt(self.risk_predictor, self.test_loader)
        print("\n ** Risk predictor model AUC:%.2f"%(auc))

        testset = list(self.test_loader.dataset)
        if 'simulation' in self.data:
            if self.data=='simulation_spike':
                with open(os.path.join('./data/simulated_spike_data/thresholds_test.pkl'), 'rb') as f:
                    th = pkl.load(f)
                with open(os.path.join('./data/simulated_spike_data/gt_test.pkl'), 'rb') as f:
                    gt_importance = pkl.load(f)
            else:
                with open(os.path.join('./data/simulated_data/state_dataset_importance_test.pkl'),'rb') as f:
                    gt_importance = pkl.load(f)

            #For simulated data this is the last entry - end of 48 hours that's the actual outcome
            label = np.array([x[1][-1] for x in testset])
            high_risk = np.where(label==1)[0]
            if len(samples_to_analyze)==0:
                samples_to_analyze = np.random.choice(range(len(label)), len(label), replace=False)
        else:
            if self.data=='ghg':
                label = np.array([x[1][-1] for x in testset])
                high_risk = np.arange(label.shape[0])
                samples_to_analyze = np.random.choice(high_risk, len(high_risk), replace=False)


        print('\n********** Visualizing a few samples **********')
        self.risk_predictor.to(self.device)
        self.risk_predictor.eval()
        if self.data=='mimic':
            signals_to_analyze = range(0, self.timeseries_feature_size)
        elif 'simulation' in self.data:
            signals_to_analyze = range(0,3)
        elif self.data=='ghg':
            signals_to_analyze = range(0,15)

        if self.data=='ghg':
            replace_and_predict(signals_to_analyze, sensitivity_analysis, data=self.data, tvec=tvec)
        else:
            for sub_ind, sample_ID in enumerate(samples_to_analyze):
                print('Fetching importance results for sample %d' % sample_ID)
                if not os.path.exists('./examples'):
                    os.mkdir('./examples')
                if not os.path.exists(os.path.join('./examples',self.data)):
                    os.mkdir(os.path.join('./examples',self.data))

                testset = list(self.test_loader.dataset)
                signals, label_o = testset[sample_ID]
                if self.data=='mimic':
                    print('Did this patient die? ', {1: 'yes', 0: 'no'}[label_o.item()])
                self.generator.load_state_dict(
                    torch.load(os.path.join('./ckpt/%s/%s.pt' % (self.data, self.generator_type))))

                tvec = range(1, signals.shape[-1])
                imps= np.zeros((len(sub_features), signals.shape[-1]-1))#[]
                top_subfeatures = []
                top_subfeatures_scores = []
                for s, sub_group in enumerate(sub_features):
                    _, imp = self._get_feature_importance_FIT(signals, sig_ind=sub_group, n_samples=50, learned_risk=self.learned_risk)
                    imps[s,:] = imp
                top_subfeatures.append([sub_features[top] for top in np.argmax(imps, axis=0)])
                top_subfeatures_scores.append(np.min(imps, axis=0))

                with open(os.path.join(self.output_path,self.predictor_model , 'top_features_' + str(sample_ID) + '.pkl'), 'wb') as f:
                    pkl.dump({'feauture_set':top_subfeatures, 'importance':top_subfeatures_scores}, f, protocol=pkl.HIGHEST_PROTOCOL)
                selected_subgroups[sample_ID] = zip(top_subfeatures, top_subfeatures_scores)
        return selected_subgroups

    def run(self, train,n_epochs, samples_to_analyze, plot=True, **kwargs):
        """ Run feature generator experiment
        :param train: (boolean) If True, train the generators, if False, use saved checkpoints

        """
        print('running for ', len(samples_to_analyze))
        testset = list(self.test_loader.dataset)
        cv = kwargs['cv'] if 'cv' in kwargs.keys() else 0
        if train and self.generator_type!='carry_forward_generator':
           self.train(n_features=self.timeseries_feature_size, n_epochs=n_epochs)
           return 0
        else:
            check_path = glob.glob(os.path.join(self.ckpt_path,'*_generator.pt'))[0]

            if not os.path.exists(check_path):
                raise RuntimeError('No saved checkpoint for this model')

            else:
                if not 'simulation' in self.data:
                    self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path,'risk_predictor_%s.pt'%self.predictor_model)))
                    self.risk_predictor_attention.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor_attention.pt')))
                    self.risk_predictor.to(self.device).eval()
                    self.risk_predictor_attention.to(self.device).eval()
                    _, _, auc, correct_label, _ = test(self.test_loader, self.risk_predictor, self.device)

                else: #simulated data
                    if self.learned_risk:
                        self.risk_predictor.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor_%s.pt'%self.predictor_model)))
                        self.risk_predictor_attention.load_state_dict(torch.load(os.path.join(self.ckpt_path, 'risk_predictor_attention.pt')))
                        self.risk_predictor.to(self.device).eval()
                        self.risk_predictor_attention.to(self.device).eval()
                        _, _, _, auc,  _ = test_model_rt(self.risk_predictor, self.test_loader)

            print("\n ** Risk predictor model AUC:%.2f"%(auc))

            if 'simulation' in self.data:
                if self.data=='simulation_spike':
                    with open(os.path.join('./data/simulated_spike_data/thresholds_test.pkl'), 'rb') as f:
                        th = pkl.load(f)

                    with open(os.path.join('./data/simulated_spike_data/gt_test.pkl'), 'rb') as f:
                        gt_importance = pkl.load(f)#Type dmesg and check the last few lines of output. If the disc or the connection to it is failing, it'll be noted there.load(f)
                        #print(gt_importance)
                else:
                    with open(os.path.join('./data/simulated_data/state_dataset_importance_test.pkl'),'rb') as f:
                        gt_importance = pkl.load(f)


                #For simulated data this is the last entry - end of 48 hours that's the actual outcome
                label = np.array([x[1][-1] for x in testset])
                #print(label)
                high_risk = np.where(label==1)[0]
                if len(samples_to_analyze)==0:
                    samples_to_analyze = np.random.choice(range(len(label)), len(label), replace=False)
                # samples_to_analyse = [101, 48, 88, 192, 143, 166, 18, 58, 172, 132]
            else:
                # gt_importance = None
                # if self.data=='mimic':
                    # samples_to_analyse = MIMIC_TEST_SAMPLES
                if self.data=='ghg':
                    label = np.array([x[1][-1] for x in testset])
                    high_risk = np.arange(label.shape[0])
                    samples_to_analyze = np.random.choice(high_risk, len(high_risk), replace=False)

            ## Sensitivity analysis as a baseline
            signal = torch.stack([testset[sample][0] for sample in samples_to_analyze])

            #Some setting up for ghg data
            if self.data=='ghg':
                label_tch = torch.stack([testset[sample][1] for sample in samples_to_analyze])
                signal_scaled = self.patient_data.scaler_x.inverse_transform(np.reshape(signal.cpu().detach().numpy(),[len(samples_to_analyze),-1]))
                tvec = list(range(1,signal.shape[2]+1,50))
            else:
                tvec = list(range(1,signal.shape[2]+1))
                signal_scaled = signal

            nt = len(tvec)
            sensitivity_analysis = np.zeros((signal.shape))
            sensitivity_start = time.time()
            if not 'simulation' in self.data:
                if self.data=='mimic' or self.data=='ghg':
                    self.risk_predictor.train()
                    for t_ind,t in enumerate(tvec):
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyze)):
                            out[s].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:,t_ind].cpu().detach().numpy()
                        signal_t.grad.data.zero_()
                self.risk_predictor.eval()
            else:
                if not self.learned_risk:
                    grad_out = []
                    for kk,i in enumerate(samples_to_analyze):
                        sample = testset[i][0].cpu().detach().numpy()
                        gt_imp = gt_importance[i,:]
                        out = np.array([self.risk_predictor(sample,gt_imp,tt) for tt in tvec])
                        #print(out.shape, sample.shape)
                        grad_x0 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==0)*sample[0,tt] for tt in tvec])
                        grad_x1 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==1)*sample[1,tt] for tt in tvec])
                        grad_x2 = np.array([5*out[tt]*(1-out[tt])*(gt_importance[i,tt]==2)*sample[2,tt] for tt in tvec])
                        grad_out.append(np.stack([grad_x0, grad_x1, grad_x2]))
                    sensitivity_analysis = np.array(grad_out)
                else:
                    #In simulation data also get sensitivity w.r.t. a learned predictor
                    self.risk_predictor.train()
                    for t_ind, t in enumerate(tvec):
                        #print(t)
                        signal_t = torch.Tensor(signal[:,:,:t]).to(self.device).requires_grad_()
                        out = self.risk_predictor(signal_t)
                        for s in range(len(samples_to_analyze)):
                            out[s].backward(retain_graph=True)
                            sensitivity_analysis[s,:,t_ind] = signal_t.grad.data[s,:, t_ind].cpu().detach().numpy()#[:,0]
                        signal_t.grad.data.zero_()
                    self.risk_predictor.eval()

            # print('Execution time of sensitivity = %.3f'%((time.time()-sensitivity_start)/float(len(samples_to_analyze))))
            print('\n********** Visualizing a few samples **********')
            all_FIT_importance = []
            all_FO_importance = []
            all_AFO_importance = []
            all_lime_importance = []

            # self.risk_predictor.load_state_dict(torch.load(os.path.join('./ckpt',self.data,'risk_predictor.pt')))
            self.risk_predictor.to(self.device)
            self.risk_predictor.eval()
            if self.data=='mimic':
                signals_to_analyze = range(0, self.timeseries_feature_size)
            elif 'simulation' in self.data:
                signals_to_analyze = range(0,3)
            elif self.data=='ghg':
                signals_to_analyze = range(0,15)

            if self.data=='ghg':
                # Replace and Predict Experiment
                replace_and_predict(signals_to_analyze, sensitivity_analysis, data=self.data, tvec=tvec)
            else:
                lime_exp = BaselineExplainer(self.train_loader, self.valid_loader, self.test_loader,
                                             self.feature_size, data_class=self.patient_data,
                                             data=self.data, baseline_method='lime')
                importance_labels = {}
                for sub_ind, sample_ID in enumerate(samples_to_analyze):
                    print('Fetching importance results for sample %d' % sample_ID)

                    lime_imp = lime_exp.run(train=train, n_epochs=n_epochs, samples_to_analyze=[sample_ID])
                    all_lime_importance.append(lime_imp)

                    top_FCC, importance, top_occ, importance_occ, top_occ_aug, importance_occ_aug, top_SA, importance_SA = self.plot_baseline(
                        sample_ID, signals_to_analyze, sensitivity_analysis[sub_ind, :, :], data=self.data, plot=plot, 
                        gt_importance_subj=gt_importance[sample_ID] if 'simulation' in self.data else None,lime_imp=lime_imp,tvec=tvec,cv=cv)
                    all_FIT_importance.append(importance)
                    all_AFO_importance.append(importance_occ_aug)
                    all_FO_importance.append(importance_occ)

                    top_signals = 4

                    FIT = []
                    for ind, sig in top_FCC[0:top_signals]:
                        imp_t = importance[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        FIT.append((i_max, t_max, max_val))
                    importance_labels.update({'FIT':FIT})

                    FO = []
                    for ind, sig in top_occ[0:top_signals]:
                        imp_t = importance_occ[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        FO.append((i_max, t_max, max_val))
                    importance_labels.update({'FO': FO})

                    SA = []
                    for ind, sig in top_SA[0:top_signals]:
                        imp_t = importance_SA[ind, 1:]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        SA.append((i_max, t_max, max_val))
                    importance_labels.update({'SA': SA})

                    AFO = []
                    for ind, sig in top_occ_aug[0:top_signals]:
                        imp_t = importance_occ_aug[ind, :]
                        t_max = int(np.argmax(imp_t.reshape(-1)))
                        i_max = int(ind)
                        max_val = float(max(imp_t.reshape(-1)))
                        AFO.append((i_max, t_max,max_val))
                    importance_labels.update({'AFO': AFO})

                    with open('./examples/%s/baseline_importance_sample_%d.json'%(self.data, sample_ID),'w') as f:
                        json.dump(importance_labels, f)
        return np.array(all_FIT_importance), np.array(all_AFO_importance), np.array(all_FO_importance), np.array(all_lime_importance), sensitivity_analysis

    def train(self, n_epochs, n_features):
        train_start_time = time.time()
        if 'joint' in self.generator_type:
            train_joint_feature_generator(self.generator, self.train_loader, self.valid_loader,
                                          generator_type=self.generator_type, n_epochs=n_epochs)
        else:
            for feature_to_predict in range(0, n_features):
                print('**** training to sample feature: ', feature_to_predict)
                self.generator = FeatureGenerator(self.feature_size, hidden_size=self.generator_hidden_size,
                                                  prediction_size=self.prediction_size, data=self.data,
                                                  conditional=self.conditional).to(self.device)
                train_feature_generator(self.generator, self.train_loader, self.valid_loader, self.generator_type,
                                        feature_to_predict, n_epochs=n_epochs)
        print("Training time for %s = " % (self.generator_type), time.time() - train_start_time)


    def plot_baseline(self, subject, signals_to_analyze, sensitivity_analysis_importance, retain_style=False, plot=False,  n_important_features=3,data='mimic',gt_importance_subj=None,lime_imp=None,tvec=None,**kwargs):
        """ Plot importance score across all baseline methods
        :param subject: ID of the subject to analyze
        :param signals_to_analyze: list of signals to include in importance analysis
        :param sensitivity_analysis_importance: Importance score over time under sensitivity analysis for the subject
        :param retain_style: Plotting mode. If true, top few important signal names will be plotted at every time point. Only true for MIMIC
        :param n_important_features: Number of important signals to plot
        """
        if not os.path.exists('./examples'):
            os.mkdir('./examples')
        if not os.path.exists(os.path.join('./examples',data)):
            os.mkdir(os.path.join('./examples',data))

        testset = list(self.test_loader.dataset)
        signals, label_o = testset[subject]
        if data=='mimic':
            print('Did this patient die? ', {1: 'yes', 0: 'no'}[label_o.item()])
        
        tvec = range(1,signals.shape[1])
        importance = np.zeros((self.timeseries_feature_size, len(tvec)))
        attention_importance = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk = np.zeros((self.timeseries_feature_size, len(tvec)))
        importance_occ = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk_occ = np.zeros((self.timeseries_feature_size, len(tvec)))
        importance_occ_aug = np.zeros((self.timeseries_feature_size, len(tvec)))
        std_predicted_risk_occ_aug = np.zeros((self.timeseries_feature_size, len(tvec)))
        max_imp_FCC = []
        max_imp_occ = []
        max_imp_occ_aug = []
        max_imp_sen = []
        AFO_exe_time = []
        FO_exe_time = []
        FIT_exe_time = []

        for i, sig_ind in enumerate(signals_to_analyze):

            if not self.generator_type=='carry_forward_generator':
                if 'joint' in self.generator_type:
                    self.generator.load_state_dict(
                        torch.load(os.path.join('./ckpt/%s/%s.pt'%(self.data, self.generator_type))))
                else:
                    if data=='mimic':
                        self.generator.load_state_dict(
                        torch.load(os.path.join(self.ckpt_path,'%s_%s.pt' % (feature_map_mimic[sig_ind], self.generator_type))))
                    elif data=='simulation':
                        self.generator.load_state_dict(
                        torch.load(os.path.join(self.ckpt_path,'%s_%s.pt' % (str(sig_ind), self.generator_type))))
                    elif data=='ghg':
                        self.generator.load_state_dict(
                        torch.load(os.path.join(self.ckpt_path,'%s_%s.pt' % (str(sig_ind),self.generator_type))))

            t0 = time.time()
            label, importance[i,:] = self._get_feature_importance_FIT( signals, sig_ind=[sig_ind], n_samples=10, learned_risk=self.learned_risk)
            t1 = time.time()
            FIT_exe_time.append(t1-t0)

            t0 = time.time()
            _, importance_occ[i, :], _, std_predicted_risk_occ[i, :] = self._get_feature_importance(signals,
                                                                                                    sig_ind=sig_ind,
                                                                                                    n_samples=10,
                                                                                                    mode="feature_occlusion",
                                                                                                    learned_risk=self.learned_risk,tvec=tvec)
            FO_exe_time.append(time.time()-t0)

            t0 = time.time()
            _, importance_occ_aug[i, :], _, std_predicted_risk_occ_aug[i, :] = self._get_feature_importance(signals,
                                                                                                            sig_ind=sig_ind,
                                                                                                            n_samples=10,
                                                                                                            mode='augmented_feature_occlusion',
                                                                                                            learned_risk=self.learned_risk,tvec=tvec)
            attention_importance[i,:] = self.risk_predictor_attention.get_attention_weights(signals.unsqueeze(0)).detach().cpu().numpy()[0,1:].reshape(-1,)
            AFO_exe_time.append(time.time()-t0)
            max_imp_FCC.append((i, max(importance[i, :])))
            max_imp_occ.append((i, max(importance_occ[i, :])))
            max_imp_occ_aug.append((i, max(importance_occ_aug[i, :])))
            max_imp_sen.append((i, max(sensitivity_analysis_importance[i, :])))

        # print('Execution time of FIT for subject %d = %.3f +/- %.3f'%(subject, np.mean(np.array(FIT_exe_time)), np.std(np.array(FIT_exe_time))) )
        # print('Execution time of AFO for subject %d = %.3f +/- %.3f' % (subject, np.mean(np.array(AFO_exe_time)), np.std(np.array(AFO_exe_time))))
        # print('Execution time of FO for subject %d = %.3f +/- %.3f' % (subject, np.mean(np.array(FO_exe_time)), np.std(np.array(FO_exe_time))))

        if 'cv' in kwargs.keys():
            cv = kwargs['cv']
        else:
            cv= 10
        if self.data=='mimic':

            sensitivity_analysis_importance = sensitivity_analysis_importance[:self.timeseries_feature_size, 1:]
        else:
            sensitivity_analysis_importance = sensitivity_analysis_importance[:,1:]

        if not os.path.exists(os.path.join(self.output_path,self.predictor_model)):
            os.mkdir(os.path.join(self.output_path,self.predictor_model))
        with open(os.path.join(self.output_path,self.predictor_model,'results_'+str(subject)+ 'cv_' + str(cv) + '.pkl'), 'wb') as f:
            pkl.dump({'FFC': {'imp':importance,'std':std_predicted_risk}, 'Suresh_et_al':{'imp':importance_occ,'std':std_predicted_risk_occ},
                      'AFO': {'imp':importance_occ_aug,'std': std_predicted_risk_occ_aug}, 'Sens': {'imp': sensitivity_analysis_importance,'std':[]},
                      'lime':{'imp':lime_imp, 'std':[]}, 'attention':{'imp': attention_importance,'std':[]}, 'gt':gt_importance_subj}, f,protocol=pkl.HIGHEST_PROTOCOL)

        if not plot:
            return max_imp_FCC, importance, max_imp_occ, importance_occ, max_imp_occ_aug, importance_occ_aug, max_imp_sen, sensitivity_analysis_importance

        max_imp_FCC.sort(key=lambda pair: pair[1], reverse=True)
        max_imp_occ.sort(key=lambda pair: pair[1], reverse=True)
        max_imp_occ_aug.sort(key=lambda pair: pair[1], reverse=True)
        max_imp_sen.sort(key=lambda pair: pair[1], reverse=True)

        max_imps = np.stack([max_imp_FCC, max_imp_occ_aug, max_imp_occ, max_imp_sen, max_imp_sen], axis=0)
        imps = np.stack([importance, importance_occ_aug, importance_occ, np.abs(sensitivity_analysis_importance), attention_importance], axis=0)
        std_imps = np.stack([std_predicted_risk, std_predicted_risk_occ_aug, std_predicted_risk_occ, np.zeros(std_predicted_risk.shape), np.zeros(std_predicted_risk.shape)], axis=0)
        n_feats_to_plot = min(self.timeseries_feature_size, n_important_features)

        plot_importance(subject, signals, label, imps, std_imps, max_imps, n_feats_to_plot, signals_to_analyze, color_map, feature_map[data],
                        data, gt_importance_subj, os.path.join(self.output_path,self.predictor_model), self.patient_data)
        return max_imp_FCC, importance, max_imp_occ, importance_occ, max_imp_occ_aug, importance_occ_aug, max_imp_sen, sensitivity_analysis_importance

    def _get_feature_importance_FIT(self, signal, sig_ind, n_samples=10, learned_risk=True, tvec=None, at_time=None, conditional=False):
        self.generator.eval()
        risks = []
        importance_FIT = []
        if tvec is None:
            tvec = range(1,signal.shape[1])

        if at_time is None:
            for t in tvec:
                if 'simulation' in self.data:
                    if not learned_risk:
                        risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
                    else:
                        risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t+1)).item()
                else:
                    risk = self.risk_predictor(signal[:,0:t+self.generator.prediction_size].view(1, signal.shape[0], t+self.generator.prediction_size)).item()

                generator_predicted_risks = []
                for _ in range(n_samples):
                    predicted_signal_FIT = signal[:,0:t+1].clone()
                    if conditional:
                        x_hat_t = self.generator.forward_conditional(signal[:, :t].unsqueeze(0), signal[:, t], [i for i in range(len(signal)) if i!=sig_ind])
                        predicted_signal_FIT[:, t] = x_hat_t
                    else:
                        x_hat_t = self.generator.forward_joint(signal[:, max(0,t-5):t].unsqueeze(0))
                        predicted_signal_FIT[:,t][sig_ind] = x_hat_t[0, sig_ind].detach().cpu()

                    if 'simulation' in self.data and not learned_risk:
                        predicted_risk_FIT = self.risk_predictor(predicted_signal_FIT.cpu().detach().numpy(), t)
                    else:
                        predicted_risk_FIT = self.risk_predictor(predicted_signal_FIT.unsqueeze(0).to(self.device)).item()
                    generator_predicted_risks.append(predicted_risk_FIT)

                # KL divergence
                probability_subsection_FIT = torch.Tensor([1-np.mean(generator_predicted_risks), np.mean(generator_predicted_risks)])
                probability_all = torch.Tensor([(1-risk), risk])

                div_FIT = torch.nn.KLDivLoss()(torch.log(torch.Tensor(probability_all)).squeeze(0),torch.Tensor(probability_subsection_FIT).squeeze(0))
                importance_FIT.append(div_FIT)
                risks.append(risk)
            return risks, importance_FIT
        else:
            if 'simulation' in self.data:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(), at_time)
                else:
                    risk = self.risk_predictor(signal[:, 0:at_time + 1].view(1, signal.shape[0], at_time + 1)).item()
            else:
                risk = self.risk_predictor(signal[:, 0:at_time + self.generator.prediction_size]
                                           .view(1, signal.shape[0], at_time + self.generator.prediction_size)).item()

            conditional_predicted_risks = []
            for _ in range(n_samples):
                # Replace signal with random sample from the distribution if feature_occlusion==True,
                # else use the generator model to estimate the value
                x_hat_t_cond, _ = self.generator.forward_conditional(signal[:, :at_time].unsqueeze(0), signal[:, at_time], sig_ind)
                predicted_signal_conditional = signal[:, 0:at_time + 1].clone()
                predicted_signal_conditional[:, -1] = x_hat_t_cond

                if 'simulation' in self.data and not learned_risk:
                    conditional_predicted_risk = self.risk_predictor(
                        predicted_signal_conditional.cpu().detach().numpy(), at_time)
                else:
                    conditional_predicted_risk = self.risk_predictor(
                        predicted_signal_conditional.unsqueeze(0).to(self.device)).item()
                conditional_predicted_risks.append(conditional_predicted_risk)

            # KL divergence
            probability_subsection = torch.Tensor([1 - np.mean(conditional_predicted_risk), np.mean(conditional_predicted_risk)])
            probability_all = torch.Tensor([(1 - risk), risk])
            div = (probability_all * (probability_all / probability_subsection).log()).sum()
            return div

    def _get_feature_importance(self, signal, sig_ind, n_samples=10, mode="feature_occlusion", learned_risk=True, tvec=None):
        self.generator.eval()
        feature_dist_0 = (np.array(self.feature_dist_0[:, sig_ind, :]).reshape(-1))
        feature_dist_1 = (np.array(self.feature_dist_1[:, sig_ind, :]).reshape(-1))

        risks = []
        importance = []
        mean_predicted_risk = []
        std_predicted_risk = []
        if tvec is None:
            tvec = range(1,signal.shape[1])
        for t in tvec:
            if 'simulation' in self.data:
                if not learned_risk:
                    risk = self.risk_predictor(signal.cpu().detach().numpy(), t)
                else:
                    risk = self.risk_predictor(signal[:, 0:t + 1].view(1, signal.shape[0], t+1)).item()
            else:
                risk = self.risk_predictor(signal[:,0:t+self.generator.prediction_size].view(1, signal.shape[0], t+self.generator.prediction_size)).item()
            signal = signal.to(self.device)
            predicted_risks = []
            for _ in range(n_samples):
                # Replace signal with random sample from the distribution if feature_occlusion==True,
                # else use the generator model to estimate the value
                if mode=="feature_occlusion":
                    prediction = torch.Tensor(np.array([np.random.uniform(-3,+3)]).reshape(-1)).to(self.device)
                    predicted_signal = signal[:, 0:t + self.generator.prediction_size].clone()
                    predicted_signal[:, t:t + self.generator.prediction_size] = torch.cat((signal[:sig_ind,
                                                                                           t:t + self.generator.prediction_size],
                                                                                           prediction.view(1, -1),
                                                                                           signal[sig_ind + 1:,
                                                                                           t:t + self.generator.prediction_size]),
                                                                                          0)
                elif mode=="augmented_feature_occlusion":
                    if self.risk_predictor(signal[:,0:t].view(1, signal.shape[0], t)).item() > 0.5:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_0)).reshape(-1,)).to(self.device)
                    else:
                        prediction = torch.Tensor(np.array(np.random.choice(feature_dist_1)).reshape(-1,)).to(self.device)
                    predicted_signal = signal[:, 0:t + self.generator.prediction_size].clone()
                    predicted_signal[:, t:t + self.generator.prediction_size] = torch.cat((signal[:sig_ind,
                                                                                           t:t + self.generator.prediction_size],
                                                                                           prediction.view(1, -1),
                                                                                           signal[sig_ind + 1:,
                                                                                           t:t + self.generator.prediction_size]),
                                                                                          0)

                if 'simulation' in self.data:
                    if not learned_risk:
                        predicted_risk = self.risk_predictor(predicted_signal.cpu().detach().numpy(), t)
                    else:
                        predicted_risk = self.risk_predictor(predicted_signal[:,0:t+self.generator.prediction_size].view(1,predicted_signal.shape[0],t+self.generator.prediction_size).to(self.device)).item()
                else:
                    predicted_risk = self.risk_predictor(predicted_signal[:, 0:t + self.generator.prediction_size].view(1, predicted_signal.shape[0], t + self.generator.prediction_size).to(self.device)).item()
                predicted_risks.append(predicted_risk)
            risks.append(risk)
            predicted_risks = np.array(predicted_risks)

            diff_imp = abs(predicted_risks-risk)
            mean_imp = np.mean(predicted_risks,0)
            std_imp = np.std(predicted_risks, 0)
            mean_predicted_risk.append(mean_imp)
            std_predicted_risk.append(std_imp)
            importance.append(np.mean(diff_imp))
        return risks, importance, mean_predicted_risk, std_predicted_risk

