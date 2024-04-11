import sys, os
import torch

sys.path.append(os.path.join(sys.path[0], r"../federated_learning"))
from Models import LSTM
import torchmetrics


class model_param:
    def __init__(self, **args) -> None:
        self.model_class = LSTM

        self.input_size = [145, 4]
        self.output_size =  1
        self.LSTM_layers = [16, 64]
        self.LSTM_dropouts = [0.1 for _ in self.LSTM_layers]
        self.LSTM_activations = [torch.nn.ReLU() for _ in self.LSTM_layers]
        self.Linear_layers = [64, 32]
        self.Linear_dropouts = [0.1 for _ in self.Linear_layers]
        self.Linear_activations = [torch.nn.ReLU() for _ in self.Linear_layers]

        # Federated parameters
        self.n_round = 700
        self.batch_size = 256 
        self.epochs = 14
        self.lr = 0.0008
        self.mu = 8
        self.device = "gpu"
        self.metric = torchmetrics.MeanSquaredError()

        # Finetuning parameters
        self.batch_size_ft = 256 
        self.epochs_ft = 350
        self.lr_ft = 0.0008
        self.metric_ft = torchmetrics.MeanSquaredError()

        key_list = ["n_round", "batch_size", "epochs", "lr" , "mu" , "input_size" , "output_size" , "LSTM_layers" ,  \
                    "LSTM_dropouts","LSTM_activations", "Linear_layers", "Linear_dropouts","Linear_activations" ,  "device" ]
        for key in key_list:
            if key in args.keys():
                setattr(self, key, args[key])
        
        #Model arg dictionnaries
        self.model_args = {"n_round" : self.n_round, "batch_size" : self.batch_size, "epochs" : self.epochs, "lr" :  self.lr, "mu" : self.mu,  \
                        "input_size" : self.input_size, "output_size" : self.output_size, "LSTM_layers" : self.LSTM_layers, \
                        "LSTM_dropouts": self.LSTM_dropouts, "LSTM_activations": self.LSTM_activations, "Linear_layers": self.Linear_layers,\
                        "Linear_dropouts" : self.Linear_dropouts, "Linear_activations" : self.Linear_activations,  "device" : self.device,\
                        "gpu" : True
                    }
        
        self.model_args_ft = {"batch_size" : self.batch_size_ft, "epochs" : self.epochs_ft, "lr" :  self.lr_ft, "input_size" : self.input_size,\
                            "output_size" : self.output_size, "LSTM_layers" : self.LSTM_layers,  "LSTM_dropouts": self.LSTM_dropouts, \
                            "LSTM_activations": self.LSTM_activations, "Linear_layers": self.Linear_layers, "Linear_dropouts" : self.Linear_dropouts,
                            "Linear_activations" : self.Linear_activations,  "device" : self.device, "gpu" : True
                    }

        self.model = self.model_class(**self.model_args)

        # Federated logging
        self.evaluation_args = [
            {"evaluation_mode" : "global_model_local_training_loss", "key":"train_loss", "reset" : True}, 
            {"evaluation_mode" : "global_model_local_training_metric", "key":"train_mse", "metric": torchmetrics.MeanSquaredError(), "reset" : True},
            {"evaluation_mode" : "aggregated_validation", "key": "federated_valid_mse", "reset": True, "metric": torchmetrics.MeanSquaredError()}
            ] 

        self.patience = 30
        self.min_epoch = 20

        self.earlystopping = {"monitor": "federated_valid_mse", "patience":self.patience, "min_delta": 0, "start_from_round":self.min_epoch,\
                              "best_weight": True}

        
    
    def reset_model(self):
        self.model = self.model_class(**self.model_args)

        
    #Client training logging
    def log_arg_client(self, client, metric):
        return {"train_loss" : {"on_epochs": False, "evaluation" : False, "on_batch": True, "metric" : client.model.loss, "show_on_progress_bar" : True,\
                                 "log": False}, 
                "validation_loss": {"on_epochs": True, "evaluation" : True, "on_batch": False, "metric" : metric, "show_on_progress_bar" : True, \
                                    "log": True} }
    

model = model_param()