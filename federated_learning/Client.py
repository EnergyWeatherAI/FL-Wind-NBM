import copy
from utils import logs, NoStdStreams

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from Models import trainer

import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError
from torchinfo import summary

"""This class represent a client for federated learning. It contains it data (with train/valid/test spliting), 
a local model to train and a copy of the global model"""
class Client(): 
    def __init__(self, name = None, **kwargs):
        self.name = name
        self.__name__ = name
        self.logs = logs()
        self._X = None
        self._y = None
        self.n_data_train = 0

        # hyperparameter for data preparation
        default_parameter = {"test_size" : 0.2, "shuffle": True, "batch_size" : 256}
        for key in default_parameter.keys():
            kwargs[key] =  kwargs[key] if key in kwargs.keys() else default_parameter[key]
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        # Models
        self.global_model = None
        self.model = None

    def __name__(self):
        return self.name
    def __repr__(self):
        return f"Client {self.__name__}"
    def __str__(self):
        return f"Client {self.__name__}"

    def prepare_model(self, model_class, **kwargs):

        default_dict = {"epochs":1, "batch_size":256, "optimizer": optim.Adam, "lr":0.001, "loss": F.mse_loss, "metrics": [MeanSquaredError], \
            "verbose":True, "gpu": False}
        self.model_class = model_class
       
        self.model_args = {}
        for key in default_dict.keys(): 
            kwargs[key] = default_dict[key] if key not in kwargs.keys() else kwargs[key]
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
            self.model_args[key] = kwargs[key]
        
        if self.verbose:
            self.model = model_class(**self.model_args)
            self.global_model = model_class(**self.model_args)
            if "state_dict" in kwargs.keys():
                self.model.load_state_dict(kwargs["state_dict"])
                self.global_model.load_state_dict(kwargs["state_dict"])
        else: 
            with NoStdStreams():
                self.model = model_class(**self.model_args)
                self.global_model = model_class(**self.model_args)
                if "state_dict" in kwargs.keys():
                    self.model.load_state_dict(kwargs["state_dict"])
                    self.global_model.load_state_dict(kwargs["state_dict"])

    def prepare_data(self, time_series= True, timestamps = None, **split_args):
        if time_series:
            self.shuffle = False
            
        self._X_train, self._X_valid, self._y_train, self._y_valid, self._idx_train, self._idx_valid =\
              train_test_split(self._X, self._y, np.arange(len(self._X)),test_size=self.test_size, shuffle=self.shuffle)
        
        self._dataset_train = MyDataset(self._X_train, self._y_train)
        self.n_data_train = len(self._dataset_train)
        self._dataset_valid = MyDataset(self._X_valid, self._y_valid)
        if timestamps is not None:
            self.timestamps_train, self.timestamps_valid = timestamps[self._idx_train], timestamps[self._idx_valid]

        self._X_test = None
        self._y_test = None
        self.timestamps_test = None
       
    def model_info(self):
        summary(self.model)

    def global_model_info(self):
        summary(self.global_model)
    
    def train(self, **kwargs): 

        # Preparing parameters
        verbose = kwargs["verbose"] if "verbose" in kwargs.keys() else False
                    
        # Change the necessary model arguments
        for key in kwargs.keys():
            if key in self.model_args.keys():
                self.model_args[key] = kwargs[key]
                setattr(self, key, kwargs[key])

        # Logging arguments if needed
        if "log_args" in kwargs.keys():
            log_args = kwargs["log_args"]
        else:
            log_args = None

        # Start training
        if verbose:
            print("  Training {} for up to {} epochs".format(self.name, self.epochs) + " "*50, end = "\r")
        self._data_loader = DataLoader(self._dataset_train, batch_size = self.batch_size)
        if verbose: 
            tr = trainer(self.model, epochs = self.epochs, batch_size = self.batch_size, device= self.model.device, progress_bar = True, \
                         evaluation_on_epoch = True, evaluation_batch_every_n = max(len(self._dataset_train)//(self.batch_size*100), 1)) 
            tr.train(self._dataset_train,  validation_dataset = self._dataset_valid, log_args =log_args, verbose = True)
            self.logs = tr.logs
        else:
            tr = trainer(self.model, epochs = self.epochs, batch_size = self.batch_size, device = self.model.device, progress_bar = False, \
                         evaluation_on_epoch = False, evaluation_batch_every_n = max(len(self._dataset_train)//(self.batch_size*100), 1)) 
            tr.train(self._dataset_train, validation_dataset = self._dataset_valid, verbose = False, log_args =log_args)
            self.logs = copy.deepcopy(tr.logs)
    
    def evaluation(self, evaluation_mode, model = None, **kwargs):

        if model is None: 
            model = self.model
        device = next(self.model.parameters()).device
        
        if evaluation_mode == "training_loss": 
            return self.loss(model(self._X_train.to(device)), self._y_train.to(device)).to("cpu")
        
        if evaluation_mode == "training_metrics":
            assert "metric" in kwargs.keys(), print("Need to pass a metric")
            return kwargs["metric"](model(self._X_train.to(device)).to("cpu"), self._y_train.to(device).to("cpu")).to("cpu")
          
        if evaluation_mode == "validation":
            metric =  kwargs["metric"] if "metric" in kwargs.keys() else self.loss
            return metric(model(self._X_train.to(device)).to("cpu"), self._y_train.to(device).to("cpu")).to("cpu")
        


class MyDataset(Dataset):
    def __init__(self, data = [], targets = [], transform_x=None, transform_y = None):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        if len(self.data.shape) == 1:
            self.data = self.data.unsqueeze(dim=1)
        
        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(dim=1)
        
        self.transform_x = transform_x
        self.transform_y = transform_y

    def concat(self, new_data): 
        self.data = torch.cat([self.data, new_data.data])   
        self.targets = torch.cat([self.targets, new_data.targets])     

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform_x:
            x = self.transform_x(x)
        if self.transform_y:
            y = self.transform_y(y)
        
        return x, y
    
    def __len__(self):
        return len(self.data)