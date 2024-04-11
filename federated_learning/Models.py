import torch
import os
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError
import pickle
from tqdm import tqdm
from utils import logs
import torchinfo 
import matplotlib.pyplot as plt
from time import time
import copy



class LSTM(nn.Module):
    def __init__(self, **kwargs):
        
        super().__init__()
        # Model default
         
        default_dict = {"optimizer":optim.Adam, "lr":0.001, "optimizer_params": {}, "loss":F.mse_loss, "metrics":[MeanSquaredError()],\
                        "name" : "model", "PATH" : r"temp_model", "gpu": True, "dropout" : True, "verbose": False}
        
               
        self.param_dict = {}       
        #Setting the parameter either from kwrags or by default
        for key in default_dict.keys(): 
            kwargs[key] = kwargs[key] if key in kwargs.keys() else default_dict[key]
        
        for key in [key for key in kwargs.keys() if key != "state_dict"]:
            setattr(self, key, kwargs[key])
            self.param_dict[key] = kwargs[key]                

        self.device = torch.device("cuda:0") if torch.cuda.is_available() and self.gpu else torch.device("cpu")
        if self.verbose:
            print("Using cuda:0" if torch.cuda.is_available() and self.gpu else "Using cpu")

        # Network architecture parameters
        network_keys = ["input_size", "output_size", "LSTM_layers", "LSTM_dropouts", "LSTM_activations", "Linear_layers", "Linear_dropouts",\
                        "Linear_activations"]
        self.network_dict = {}
        for key in network_keys:
            assert key in kwargs.keys(), print(f"Need {key} parameters")
            self.network_dict[key] = copy.deepcopy(kwargs[key])
            
        self.init_network()
    
    def init_network(self):
        # Network Layer, the variable network_param determines the archicteture 
        
        setattr(self, f"LSTM_layer_{0}", nn.LSTM(input_size=self.input_size[1], hidden_size = self.LSTM_layers[0], batch_first=True))
        setattr(self, f"LSTM_dropout_{0}", nn.Dropout(p= self.LSTM_dropouts[0]))
        setattr(self, f"LSTM_activation_{0}", self.LSTM_activations[0])
        
        for i in range(len(self.LSTM_layers)-1):
            setattr(self, f"LSTM_layer_{i+1}", nn.LSTM(input_size=self.LSTM_layers[i], hidden_size = self.LSTM_layers[i+1], batch_first=True))
            setattr(self, f"LSTM_dropout_{i+1}", nn.Dropout(p= self.LSTM_dropouts[i+1]))
            setattr(self, f"LSTM_activation_{i+1}", self.LSTM_activations[i+1])      

        dim = self.LSTM_layers[-1]

        setattr(self, f"Linear_layer_{0}", nn.Linear(in_features=dim, out_features = self.Linear_layers[0]))
        setattr(self, f"Linear_dropout_{0}", nn.Dropout(p= self.LSTM_dropouts[0]))
        setattr(self, f"Linear_activation_{0}", self.Linear_activations[0])
        
        for i in range(0,len(self.Linear_layers)-1):
            setattr(self, f"Linear_layer_{i+1}", nn.Linear(in_features=self.Linear_layers[i], out_features = self.Linear_layers[i+1]))
            setattr(self, f"Linear_dropout_{i+1}", nn.Dropout(p= self.LSTM_dropouts[0]))
            setattr(self, f"Linear_activation_{i+1}", self.Linear_activations[i+1])
                    
        setattr(self, f"Linear_layer_{len(self.Linear_layers)}", nn.Linear(in_features=self.Linear_layers[-1], out_features = self.output_size)) 
       
    def forward(self, x):
        for i in range(len(self.LSTM_layers)):
            x, _ = getattr(self, f"LSTM_layer_{i}")(x)
            if self.dropout:
                x = getattr(self, f"LSTM_dropout_{i}")(x)
            x = getattr(self, f"LSTM_activation_{i}")(x)

        x = x[:,-1,:]

        for i in range(len(self.Linear_layers)):
            x = getattr(self, f"Linear_layer_{i}")(x)
            if self.dropout:
                x = getattr(self, f"Linear_dropout_{i}")(x)
            x = getattr(self, f"Linear_activation_{i}")(x)

        x = getattr(self, f"Linear_layer_{i+1}")(x)
        return x
       
    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr, **self.optimizer_params)
        return optimizer
    
    def summary(self):
        dim = (1, *self.input_size)
        x = torch.randn(dim, device = "cpu")
        self(x)
        torchinfo.summary(self, dim, device="cpu", col_names = ("input_size", "output_size", "num_params"), verbose = 1)



class trainer:
    def __init__(self, model, **kwargs):
        self.model = model
        self.param_dict = {"epochs": 1, "batch_size": 256, "lr": self.model.lr, "optimizer_params":model.optimizer_params, "device" : model.device, \
                           "progress_bar": False, "display_every_n" : (-1, None, None), "evaluation_on_epoch" : True, "evaluation_batch_every_n" : 1, \
                            "num_workers" : 0}  

        for key in self.model.param_dict.keys():
            if key in kwargs.keys():
                self.param_dict[key] = kwargs[key]
                self.model.param_dict[key] = kwargs[key]
                setattr(self.model, key, kwargs[key])

        for key in self.param_dict.keys():
            if key in kwargs.keys():
                self.param_dict[key] = kwargs[key]
            setattr(self, key, self.param_dict[key])
            
        self.logs = logs()
        self.optimizer = self.model.configure_optimizers()
    
    def train_one_epoch(self, e, log_args): 
        running_res = {key : 0. for key in log_args.keys()}
        last_res = {key : 0. for key in log_args.keys()}

        if self.verbose:
            loop = tqdm(self.train_loader, total = len(self.train_loader), position=0, leave= True) 

        for i, data in enumerate(self.train_loader):
            
            inputs, labels = data
            inputs, labels = inputs.clone().detach().to(self.device), labels.clone().detach().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Compute the loss and its gradients
            loss = self.model.loss(outputs, labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_res["train_loss"] += loss.item()

            for key in [key for key in log_args.keys() if key != "train_loss"]:
                running_res[key] += log_args[key]["metric"](outputs, labels).item()

            if i % self.evaluation_batch_every_n == self.evaluation_batch_every_n -1:
                for key in log_args.keys():
                    last_res[key] = running_res[key] / self.evaluation_batch_every_n
                    running_res[key] = 0.
                if self.verbose:
                    loop.set_description(f"Epoch [{e+1}/{self.epochs}]")
                    loop.set_postfix({key : last_res[key] for key in last_res.keys() if log_args[key]["show_on_progress_bar"]}) 
            if self.verbose:
                loop.update(1)
        del loss, outputs
        return last_res
    
    def train(self, train_dataset,   **kwargs):  
        
        # Preparing the datasets
        self.train_dataset = train_dataset
        if "validation_dataset" in kwargs.keys():
            self.validation_dataset = kwargs["validation_dataset"] 
            self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers= self.num_workers)
            self.validation_loader = DataLoader(self.validation_dataset, batch_size = self.batch_size, shuffle = True, num_workers= self.num_workers)
        else:
            self.train_loader = DataLoader(self.train_dataset, batch_size = self.batch_size, shuffle = True, num_workers= self.num_workers)
            self.validation_dataset = None
            self.validation_loader = None 

        
        # Default parameters
        default_dict = {"save_best_model" : False, "best_save_path" : r".temp",\
            "time_log" : False, "verbose": True, "gpu":True}
        
        for key in default_dict.keys():
            setattr(self, key, kwargs[key] if key in kwargs.keys() else default_dict[key])

        self.device = torch.device("cuda:0") if torch.cuda.is_available() and self.gpu else torch.device("cpu")
        if self.verbose:
            print("Using cuda:0" if torch.cuda.is_available() and self.gpu else "Using cpu")
           
        self.model.to(self.device)
        #self.train_loader = DataLoader(self.train_loader, self.device)
        #if self.validation_loader is not None:
        #    self.validation_loader =  DataLoader(self.validation_loader, self.device)            

        # Setup of log_args dict
        log_args = {"train_loss" : {"on_epochs": False, "evaluation" : False, "on_batch": True, "metric" : self.model.loss, "show_on_progress_bar" : True, \
                                    "log": False}}
        if self.validation_dataset is not None:
            log_args["validation_loss"] = {"on_epochs": True, "evaluation" : True, "on_batch": False, "metric" : self.model.loss,\
                                             "show_on_progress_bar" : True, "log": True}
        
        if "log_args" in kwargs.keys() and kwargs["log_args"] is not None:
            for key in kwargs["log_args"].keys():
                log_args[key] = kwargs["log_args"][key]  
       
        # Setup earlystopping 
        ES = kwargs["earlystopping"] if "earlystopping" in kwargs.keys() and kwargs["earlystopping"] is not None else False 
        if ES:
            earlystopping_dict = kwargs["earlystopping"]
            default_param = {"patience":1, "min_delta": 0, "best_weight": False, "start_from_round":0, "baseline": None, "verbose_earlystopping":True}
            assert "monitor" in earlystopping_dict.keys(), print("Need a quantity to monitor")
            assert earlystopping_dict["monitor"] in [key for key in log_args.keys() if log_args[key]["log"]], print("The monitored quantity must be logged")
            self.monitor_key = earlystopping_dict["monitor"]
            for key in default_param.keys():
                setattr(self, key, earlystopping_dict[key] if key in earlystopping_dict.keys() else default_param[key])
               
            last_res = None
            count = 0
      
        for e in range(self.epochs):

            if self.time_log: start = time()

            log_args_batch = {key: log_args[key] for key in log_args.keys() if log_args[key]["on_batch"]}
            self.model.train(True)

            self.train_one_epoch(e, log_args_batch) 

            # Epochs only on training set
            self.model.eval() # Remove dropout and training specific effects
            with torch.no_grad():
                for i, tdata in enumerate(self.train_loader):
                        tinputs, tlabels = tdata
                        tinputs, tlabels = tinputs.clone().detach().to(self.device), tlabels.clone().detach().to(self.device)
                        toutputs = self.model(tinputs)
                        log_args_epochs_train = {key : log_args[key] for key in log_args.keys() if log_args[key]["on_epochs"] \
                                                and not log_args[key]["evaluation"]}
                        
                        res_train = {key : 0 for  key in log_args_epochs_train.keys()}
                        for key in log_args_epochs_train.keys():
                            res_train[key] += log_args_epochs_train[key]["metric"](toutputs, tlabels) 
                
                for key in res_train.keys():
                    res_train[key] = res_train[key]/(i+1)

            # Set the model to evaluation mode, disabling dropout and using population
            if self.validation_loader is not None:
                with torch.no_grad():
                    for i, vdata in enumerate(self.validation_loader):
                        vinputs, vlabels = vdata
                        vinputs, vlabels = vinputs.clone().detach().to(self.device), vlabels.clone().detach().to(self.device)
                        voutputs = self.model(vinputs)
                        
                        log_args_epochs_eval = {key : log_args[key] for key in log_args.keys() if log_args[key]["on_epochs"] and log_args[key]["evaluation"]}
                        res_eval = {key : 0 for  key in log_args_epochs_eval.keys()}
                        for key in log_args_epochs_eval.keys():
                            if "to" in dir(log_args_epochs_eval[key]["metric"]):
                                metric = log_args_epochs_eval[key]["metric"].to(self.device)
                            else:
                                metric = log_args_epochs_eval[key]["metric"]
                            res_eval[key] += metric(voutputs.to(self.device), vlabels.to(self.device)).to(self.device)
                for key in res_eval.keys():
                    res_eval[key] = res_eval[key]/(i+1)
                

                # Add the results to logs 
                for key in [key for key in log_args.keys() if log_args[key]["log"]]:
                    if key in res_eval.keys():
                        self.logs.add_value(key, res_eval[key].to("cpu").detach().numpy())
                    elif key in res_train.keys():
                        self.logs.add_value(key, res_train[key].to("cpu").detach().numpy())

            if self.time_log:
                end = time()
                self.logs.add_value("time", end-start)   

            if self.verbose:
                res = res_train | res_eval
                for key in res.keys():
                    print(f"{key} : {res[key]}", end = ", ")
                print("")

            # Track best performance, and save the model's state
            if self.save_best_model:
                if "validation_loss" in res_train.keys():
                    if e == 0:
                        best_loss = res_train["validation_loss"]
                    else:
                        if res_train["validation_loss"] < best_loss:
                            best_loss = res_train["validation_loss"]
                            model_path = os.path.join(self.best_save_path, f"best_{self.model.name}.pth")
                            torch.save(self.model.state_dict(), model_path)
                else:
                    if e == 0:
                        best_loss = res_train["train_loss"]
                    else:
                        if res_train["train_loss"] < best_loss:
                            best_loss = res_train["train_loss"]
                            model_path = os.path.join(self.best_save_path, f"best_{self.model.name}.pth")
                            torch.save(self.model.state_dict(), model_path)

            if ES and e>= self.start_from_round:
                new_res = self.logs[self.monitor_key][-1]  # Must be a 1D list of values (not per client)
                
                if last_res is None:
                    last_res = new_res
                    if self.verbose_earlystopping:
                        print(f"{self.monitor_key} : {last_res}, current patience count : {count} out of {self.patience}")
                    if self.best_weight:
                        best_res = new_res
                        best_round = e
                        best_state_dict = self.model.state_dict()
                else:
                    if (self.baseline is not None and new_res < self.baseline and last_res - new_res <= self.min_delta) \
                        or (self.baseline is None and last_res - new_res <= self.min_delta):
                        count+=1

                    else:
                        count = 0
                        last_res = new_res
                    
                    if self.best_weight:
                        if new_res < best_res:
                            best_res = new_res
                            best_round = e
                            best_state_dict = self.model.state_dict()
                    if self.verbose_earlystopping:
                        print(f"{self.monitor_key} : {last_res}, current patience count : {count} out of {self.patience}")
                    if count >= self.patience:
                        if self.best_weight:
                            return best_round, best_state_dict, best_res
                        return
            if e == self.epochs-1:
                if self.validation_loader:
                    return e, self.model.state_dict(), self.logs["validation_loss"][-1]
                else:
                    return e, self.model.state_dict()    
