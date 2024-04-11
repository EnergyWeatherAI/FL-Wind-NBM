import copy
import torch
from torch import optim
import torch.nn.functional as F
from torchmetrics.regression import MeanSquaredError
from utils import logs, execution_time
import numpy as np
from scipy.special import softmax
from bayes_opt import BayesianOptimization

""" Abstract class implementing a central server for federated learning """
class Server():
    def __init__(self, model_class, client_list, **kwargs):
        """ 
        model_class : class, model architecture
        client_list : list of Client object, list of the clients
        
        **n_round : int, number of federated rounds
        **batch_size : int, training batch_size
        **epochs : int, number of epochs to train the client on
        **lr : float, learning rate of the optimizer
        **optimizer : object, optimizer of the model
        **loss : object, loss function
        **metrics : list of objects, list of metric to evaluate the model
        **verbose : Bool, decide how much to print"""

        self.name = "Server"
        self.model_class = model_class
        self.client_list = client_list
        self.logs = logs()
        self.round_number = 0

        #Setting the parameter for the model either from kwrags or by default
        default_model_dict = {"n_round" : 5, "batch_size" : 256, "epochs" : 10, "lr" : 0.001, "optimizer" : optim.Adam,
                                "loss" : F.mse_loss, "metrics" : [MeanSquaredError], "verbose": False, "gpu": False, "device": "cuda:0"}
        
        # Model parameters
        self.model_args = {}
        for key in default_model_dict.keys(): 
            kwargs[key] =  default_model_dict[key] if key not in kwargs.keys() else kwargs[key]

        for key in kwargs.keys(): 
                setattr(self, key, kwargs[key])
                self.model_args[key] = kwargs[key]

        # Initialise the global model
        self.global_model = self.model_class(**self.model_args)
        if "init_state_dict" in kwargs.keys():
            self.global_model.load_state_dict(kwargs["init_state_dict"])

    
    def initialize_clients(self): #Abstract method
        pass    

    def training_round(self): #Abstract method
        pass

    def training(self, round_number = None, show = True, **kwargs):
        """  
        round_number : int (Optional), number of federate training rounds. 
        ** evaluate_args (Optional): list of parameter dictionnaries to pass in self.evaluation
        ** early_stopping (Optional) : dictionnary of early stopping parameter (evaluation type, waiting, return_best etc...) !the metric must be a key evaluation arg!
        """ 

        if "evaluation_args" in kwargs.keys():
            self.evaluate(kwargs["evaluation_args"], first_call = True if "first_call" not in kwargs.keys() else kwargs["first_call"])

        if round_number is None:
            round_number = self.n_round
        
        self.device = "cuda:0" if self.device == "gpu" else self.device

        ES = "earlystopping" in kwargs.keys()
        if ES:
            earlystopping_dict = kwargs["earlystopping"]
            default_param = {"patience":1, "min_delta": 0, "best_weight": False, "start_from_round":0, "baseline": None, "verbose_earlystopping":True}
            assert "monitor" in earlystopping_dict.keys(), print("Need a quantity to monitor")
            assert earlystopping_dict["monitor"] in self.logs.keys(), print("The monitored quantity must be logged")
            self.monitor_key = earlystopping_dict["monitor"]
            for key in default_param.keys():
                setattr(self, key, earlystopping_dict[key] if key in earlystopping_dict.keys() else default_param[key])
            last_res = None 
            count = 0 # Patience count since last best result

        for i in range(round_number):
            self.round_number = i # Used in FedProx 
            if show:
                print("Federated Training round number {} ".format(i+1))
            self.training_round()

            if "evaluation_args" in kwargs.keys():
                    self.evaluate(kwargs["evaluation_args"])
            
            if ES and i>= self.start_from_round:
                new_res = self.logs[self.monitor_key][-1]  # Must be a 1D list of values (not per client)
        
                if last_res is None:
                    last_res = new_res
                    best_res = new_res
                    best_round = self.round_number
                    if self.verbose_earlystopping:
                        print(f"{self.monitor_key} : {last_res}, current patience count : {count} out of {self.patience}")
                    if self.best_weight:
                        best_round = i
                        best_state_dict = copy.deepcopy(self.global_model.state_dict())
                else:
                    if (self.baseline is not None and new_res < self.baseline and last_res - new_res <= self.min_delta) \
                        or (self.baseline is None and last_res - new_res <= self.min_delta):
                        count+=1
                    else:
                        count = 0
                        last_res = new_res
                    if new_res < best_res:
                        best_res = new_res 
                        if self.best_weight:
                            best_round = i
                            best_state_dict = copy.deepcopy(self.global_model.state_dict())
                    if self.verbose_earlystopping:
                        print(f"{self.monitor_key} : {new_res}, current patience count : {count} out of {self.patience}")
                    if count >= self.patience:
                        break

        # Post-training 
        if ES and self.best_weight:
            self.logs.add_log("num_round", best_round)
            self.global_model.load_state_dict(best_state_dict)
        else:
            self.logs.add_log("num_round", log = i, new = True if "first_call" not in kwargs.keys() else kwargs["first_call"])
        
        for client in self.client_list:
            client.model.load_state_dict(self.global_model.state_dict())
            client.global_model.load_state_dict(self.global_model.state_dict())
    
    def evaluate(self, list_evaluate_args, model = None, first_call = False):
        
        for evaluate_args in list_evaluate_args: 
            assert "evaluation_mode" in evaluate_args.keys(), print("Need to pass an evaluate_mode")
            if ("key" in evaluate_args.keys() and evaluate_args["key"] in self.logs.keys()) \
                and (first_call or (evaluate_args["reset"] if "reset" in evaluate_args.keys() else False)):
                self.logs.pop(evaluate_args["key"])
            with torch.no_grad():
                self.evaluation(model = model, **evaluate_args)

    def evaluation(self, evaluation_mode, model = None, logging = True,  **kwargs):
        """ 
        evaluation_mode : str describe the type of evlauation desired
              
        """
        
        if evaluation_mode == "global_model_local_training_loss":
            key = kwargs["key"] if "key" in kwargs.keys() else "global_model_local_training_loss"
            dict_training_loss = {}
            for client in self.client_list:
                # Computed by the client
                dict_training_loss[client.name] = client.evaluation(model = model, evaluation_mode= "training_loss")
            if logging:
                self.logs.add_value(key, dict_training_loss)
                self.logs[key] = self.merge_dict_list(self.logs[key])
                return dict_training_loss
            else: 
                return dict_training_loss

        if evaluation_mode == "global_model_local_training_metric":
            assert "metric" in kwargs.keys(), print("Need to pass a metric to evaluate")

            key = kwargs["key"] if "key" in kwargs.keys() else"global_model_local_training_metrics"
            dict_training_loss = {}
            
            for client in self.client_list:
                dict_training_loss[client.name] = client.evaluation(evaluation_mode= "training_metrics", model = client.model, **kwargs)
            if logging:
                self.logs.add_value(key, dict_training_loss)
                self.logs[key] = self.merge_dict_list(self.logs[key])
                return dict_training_loss
            else:
                return dict_training_loss

        if evaluation_mode == "local_model_local_validation":
            key = kwargs["key"] if "key" in kwargs.keys() else "local_model_local_validation"              
            dict_res = {}
            for client in self.client_list:
                dict_res[client.name] = client.evaluation(model= model, evaluation_mode= "validation", **kwargs)
            if logging:
                self.logs.add_value(key, dict_res)
                self.logs[key] = self.merge_dict_list(self.logs[key])
                return dict_res
            else:
                return dict_res

        if evaluation_mode == "global_aggregated_loss": 
            key = kwargs["key"] if "key" in kwargs.keys() else "global_aggregated_loss"
          
            dict_training_loss = {}
            for client in self.client_list:
                # Computed by the client
                dict_training_loss[client.name] = client.evaluation(evaluation_mode= "training_loss", model = client.model)

            # Weight to compute the agregated loss, default is uniform 
            w = [1/len(self.client_list) for _ in self.client_list]
            if "weigthed" in kwargs.keys() and kwargs["weighted"]:
                if "weights" not in kwargs.keys(): # Weight proportional to data quantity
                    N = sum([client.n_data_train for client in self.client_list])
                    w = [client.n_data_train/N for client in self.client_list]
                else: # Custom weights
                    w = kwargs["weights"]

            res = sum([dict_training_loss[client.name]*w[i] for i, client in enumerate(self.client_list)])
            if logging:
                self.logs.add_value(key, res.to("cpu").detach().numpy())
                return res
            else:
                return res
            

        if evaluation_mode == "aggregated_validation": 

            key = kwargs["key"] if "key" in kwargs.keys() else "aggregated_validation"
            dict_validation = {}
            for client in self.client_list:
                # Computed by the client
                dict_validation[client.name] = client.evaluation(evaluation_mode= "validation", **kwargs)

            # Weight to compute the agregated loss, default is uniform 
            w = [1/len(self.client_list) for _ in self.client_list]
            if "weigthed" in kwargs.keys() and kwargs["weighted"]:
                if "weights" not in kwargs.keys(): # Weight proportional to data quantity
                    N = sum([client.n_data_train for client in self.client_list])
                    w = [client.n_data_train/N for client in self.client_list]
                else: # Custom weights
                    w = kwargs["weights"]
                    
            res = sum([dict_validation[client.name]*w[i] for i, client in enumerate(self.client_list)])
            if logging:
                self.logs.add_value(key, res)
                return res
            else:
                return res
               
    @staticmethod
    def merge_dict_list(list):
        # List of dictionnaries with identical key values.
        if len(list) <= 1:
            return list
        res = {}
        for dic in list:
            for key in dic.keys():
                if key in res.keys():
                    if len(dic[key].shape)==0:
                        new = torch.unsqueeze(dic[key], dim = 0)
                    else:
                        new = dic[key].to("cpu").detach().numpy()
                    res[key] = np.concatenate([res[key], new])
                else:
                    if len(dic[key].shape)==0:
                         res[key] = torch.unsqueeze(dic[key], dim = 0).to("cpu").detach().numpy()
                    else:
                         res[key] = dic[key]
        return [res]
   
    
class FedAvg(Server):
    def __init__(self, model_class, client_list, size_weighting = True, **kwargs):
        
        super().__init__(model_class, client_list, **kwargs) # Server instance
        self.name = "FedAvg"

        # By default, FedAvg use weighted averaging proportional to the data quantity
        default_args = {"size_weighting": True}
        for key in default_args.keys(): 
            setattr(self, key, kwargs[key] if key in kwargs.keys() else default_args[key])

        self.initialize_clients()

    def initialize_clients(self): 
        for client in self.client_list:   
            client.prepare_model(self.model_class, state_dict = self.global_model.state_dict(), **self.model_args)
            client.prepare_data()  #Initialize the data for torch usage

    @execution_time()
    def training_round(self):
        # Send the aggregated model to the clients
        for client in self.client_list:
            client.model.load_state_dict(self.global_model.state_dict())
            client.global_model.load_state_dict(self.global_model.state_dict())

        # Train the models
        for client in self.client_list:
            client.train(verbose = False)

        # Getting and averaging client model
        state_dic = {}
        N = np.sum([client.n_data_train for client in self.client_list])
        for key in self.global_model.state_dict():
            if self.size_weighting:
                state_dic[key] = torch.sum(torch.stack([client.n_data_train/N * client.model.state_dict()[key] for client in self.client_list], dim = 0), dim = 0)
            else:
                state_dic[key] = torch.mean(torch.stack([client.model.state_dict()[key] for client in self.client_list], dim = 0), dim = 0)
        self.global_model.load_state_dict(state_dic)


class FedProx(Server):
    def __init__(self, model_class, client_list, mu,  size_weighting = True, prox_norm = F.mse_loss, **kwargs):
        
        super().__init__(model_class, client_list, **kwargs) # Server instance

        self.name = "FedProx"
        # Default parameters for FedProx
        default_args = {
            "mu": 1, \
            "size_weighting": True, \
            "prox_norm": F.mse_loss
        }

        for key in default_args.keys(): 
            setattr(self, key, kwargs[key] if key in kwargs.keys() else default_args[key])
            
        self.base_loss = self.model_args["loss"]  # The base training loss
        self.initialize_clients()

    def initialize_clients(self): 
        for client in self.client_list:
           
            client.prepare_model(self.model_class, state_dict = self.global_model.state_dict(), **self.model_args) 
            client.prepare_data()  
           
            # Setting up the FedProx modified loss
            parameters = client.model.parameters()
            global_parameters = client.global_model.parameters()
            if self.gpu:
                if torch.cuda.is_available():
                    self.device =  torch.device('cuda:0')
                else:
                    print("GPU unavailable")
                    self.device = torch.device("cpu")

            # Defining the new loss
            def loss(output, label):
                mse = 0
                for global_param, param in zip(global_parameters, parameters):
                    mse += F.mse_loss(param.to(self.device), global_param.to(self.device))
                base_loss = self.base_loss(output.to(self.device), label.to(self.device))
                res = base_loss + torch.ones(1, requires_grad = True, device = self.device)*self.mu*mse
                return res
                    
            # Pass the new loss
            self.model_args["loss"] = loss

            client.prepare_model(self.model_class, state_dict = self.global_model.state_dict(), **self.model_args) 
            client.prepare_data()  
  
    @execution_time()
    def training_round(self):
        # Send the aggregated model to the clients
        for client in self.client_list:
            client.model.load_state_dict(self.global_model.state_dict())
            client.global_model.load_state_dict(self.global_model.state_dict())
        
        # Train the models
        for client in self.client_list:
            client.train(verbose = False)
            torch.cuda.empty_cache()

        # Getting and averaging client model
        state_dic = {}
        N = np.sum([client.n_data_train for client in self.client_list])
        for key in self.global_model.state_dict():
            if self.size_weighting:
                state_dic[key] = torch.sum(torch.stack([client.n_data_train/N * client.model.state_dict()[key] for client in self.client_list], dim = 0), dim = 0)
            else:
                state_dic[key] = torch.mean(torch.stack([client.model.state_dict()[key] for client in self.client_list], dim = 0), dim = 0)
        self.global_model.load_state_dict(state_dic)

    
class FarmClustering(torch.nn.Module):
    def __init__(self, model_class, server_list, **kwargs):
   
        super().__init__()
        
        self.name = "FarmClustering" 
        self.logs = logs()
        self.server_list = server_list
        self.farm_names = [f"server {i+1}" for i in range(len(self.server_list))] if "farm_names" not in kwargs.keys() else kwargs["farm_names"]
        for server, name in zip(self.server_list, self.farm_names):
            setattr(server, "farm_name", name)
            
        default_param = {"gpu": False, "mix_every_n" : 1, "update_coeff_every_n_mix": 5, "init_points": 5,\
              "n_iter": 10, "verbose": 0, "acquisition_function": None}
        for key in default_param.keys():
            if key not in kwargs.keys():
                kwargs[key] = default_param[key]
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device("cpu", 0)
        
        default_model_args = {"n_round" : 50,
                            "batch_size" : 256, 
                            "epochs" : 5,
                            "lr" :  0.0002, 
                            "input_size" : torch.Size([144,4]), 
                            "output_size" :1,
                            "LSTM_layers" : [16,16], 
                            "LSTM_dropouts": [0.1,0.1],
                            "LSTM_activations": [torch.nn.ReLU(), torch.nn.ReLU()] ,
                            "Linear_layers": [512, 256], 
                            "Linear_dropouts" : [0.3, 0.3], 
                            "Linear_activations" : [torch.nn.ReLU(), torch.nn.ReLU()],
                            "device" : self.device
                            }
        self.model_args = copy.deepcopy(default_model_args)
        for key in default_model_args.keys():
            if key in kwargs.keys():
                self.model_args[key] = kwargs[key]

        self.mix_update_index = 0
        self.num_farm = len(self.server_list)
        self.coeff = {}
        for name in [server.farm_name for server in self.server_list]:
            # Initialising as intra-farm learning
            self.coeff[name] = [1. if n == name else 0. for n in [s.farm_name for s in self.server_list]]       

    @execution_time()
    def training_round(self):
    
        if self.mix_update_index>0:
            self.coeff_optim()
            self._mix_models()
        self.mix_update_index += 1

        # Train the models with fed average on each farm
        for server in self.server_list:
            print(f"Training {server.farm_name} for {self.mix_every_n} rounds")
            server.training(evaluation_args = self.evaluation_args, round_number = self.mix_every_n, first_call = True if self.mix_update_index == 0 \
                            else False, show = False)

        
        
    def _mix_models(self, coeffs = None):
        # coeffs is a dictionnaries of list
        coeffs = self.coeff if coeffs is None else coeffs
        state_dic = {}
        for server in self.server_list:
            state_dic[server.farm_name] = {}
            prob = softmax(coeffs[server.farm_name])
            for key in server.global_model.state_dict():
                state_dic[server.farm_name][key]  = torch.sum(torch.stack([prob[i]*ser.global_model.state_dict()[key] \
                                                                for i,ser in enumerate(self.server_list)], dim = 0), dim = 0)
            
        for server in self.server_list:
            server.global_model.load_state_dict(state_dic[server.farm_name])

    def evaluation(self, coeff, server):
        state_dic = {}
        new_server = copy.deepcopy(server)
        prob = softmax(coeff)
        for key in server.global_model.state_dict():
            state_dic[key] = torch.sum(torch.stack([prob[i]*ser.global_model.state_dict()[key] \
                                                            for i,ser in enumerate(self.server_list)], dim = 0), dim = 0)
            
        new_server.global_model.load_state_dict(copy.deepcopy(state_dic))
        
        device = next(new_server.global_model.parameters()).device
        
        
        r = {}
        w = [1/len(server.client_list) for _ in server.client_list] 
        for client in server.client_list:
            # need to be computed by the client
            metric = MeanSquaredError()   
            X,y = client._X_valid, client._y_valid
            r[client.name] =  metric(new_server.global_model(X.to(device)).to(device), y.to(device)).to(device)
        
        res = sum([r[client.name]*w[i] for i, client in enumerate(server.client_list)])
        return res.detach().to("cpu")

    @execution_time()
    def coeff_optim(self):
        b_optim = {}
        for server in self.server_list:
            bayesian_fn = lambda c1,c2,c3: -self.evaluation([c1,c2,c3], server)
            pbounds = {'c1': (-10, 10), 'c2': (-10, 10), 'c3': (-10, 10)}
            b_optim[server.farm_name] = BayesianOptimization(f = bayesian_fn, pbounds = pbounds, verbose = self.verbose, allow_duplicate_points=True)
            intra_point = [10. if n == server.farm_name else -10. for n in [s.farm_name for s in self.server_list]]
            target = bayesian_fn(*intra_point)

            #Register the intra farm learning as reference point
            b_optim[server.farm_name].register(params=intra_point, target=target)

            # maximize
            b_optim[server.farm_name].maximize(init_points=self.init_points, n_iter=self.n_iter, acquisition_function = self.acquisition_function)
     
            self.coeff[server.farm_name] = list(b_optim[server.farm_name].max['params'].values())
            # TODO Maybe add some verbose here.
        prob_dict = {key : softmax(self.coeff[key]) for key in self.coeff.keys()}
        prob_show =  {key : list(map(lambda x : round(x,3), softmax(self.coeff[key]))) for key in self.coeff.keys()}
        print(f"Updated coefficient give the probabilities {prob_show}")
        self.logs.add_log(key = "prob_coeff", log = prob_dict, new = True if self.mix_update_index == 0 else False)
   
            
    def train(self, round_number = None, **kwargs):
        """  
        round_number : int (Optional), number of federate training rounds. 
        ** evaluate_args (Optional): list of parameter dictionnaries to pass in self.evaluation
        ** early_stopping (Optional) : dictionnary of early stopping parameter (evaluation type, waiting, return_best etc...) !the metric must be a 
        key evaluation arg!
        """ 
        if "evaluation_args" in kwargs.keys():
            self.evaluation_args = kwargs["evaluation_args"]
            
        if round_number is None:
            round_number = self.n_round
        ES = False
        if "earlystopping" in kwargs.keys():
            self.earlystopping = kwargs["earlystopping"]
            earlystopping_dict = kwargs["earlystopping"]
            default_param = {"patience":1, "min_delta": 0, "best_weight": False, "start_from_round":0, "baseline": None, "verbose_earlystopping":True}
            assert "monitor" in earlystopping_dict.keys(), print("Need a quantity to monitor")
            assert earlystopping_dict["monitor"] in [ev['key'] for ev in self.evaluation_args], print("The monitored quantity must be logged")
            self.monitor_key = earlystopping_dict["monitor"]
            for key in default_param.keys():
                if key in earlystopping_dict.keys():
                    setattr(self, key, earlystopping_dict[key])
                else:
                    setattr(self, key, default_param[key])
            ES = True
            last_res = None
            count = 0
        for i in range(round_number):
            self.round_number = i
            print("\nGlobal round number {} \n".format(i+1))
            self.training_round()

            if "evaluation_args" in kwargs.keys():
                for server in self.server_list:
                    self.logs[f"server_{server.farm_name}"] = copy.deepcopy(server.logs)
                    
            if ES and i>= self.start_from_round:
                new_res = np.sum([len(server.client_list)*server.logs[self.monitor_key][-1] for server in self.server_list])/\
                    (np.sum([len(server.client_list) for server in self.server_list]))  # Average federated validation across the farms/clients

                
                if last_res is None:
                    last_res = new_res
                    if self.verbose_earlystopping:
                        m = kwargs["earlystopping"]["monitor"]
                        print(f"{m} : {last_res}, current patience count : {count} out of {self.patience}")
                    if self.best_weight:
                        best_res = new_res
                        best_round = i
                        best_state_dicts = {f"{server.farm_name}_best_dict":copy.deepcopy(server.global_model.state_dict()) \
                                            for server in self.server_list}
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
                            best_round = i
                            best_state_dicts = {f"{server.farm_name}_best_dict":copy.deepcopy(server.global_model.state_dict()) \
                                            for server in self.server_list}
                    if self.verbose_earlystopping:
                        m = kwargs["earlystopping"]["monitor"]
                        print(f"{m} : {last_res}, current patience count : {count} out of {self.patience}")
                    if count >= self.patience:
                        break
                    

        # log global results
        self.logs["res"] = np.sum([len(server.client_list)*torch.stack(server.logs[self.monitor_key]).detach().numpy() for server in self.server_list], axis = 0)/\
                    (np.sum([len(server.client_list) for server in self.server_list]))
        if ES and self.best_weight and i > 2:
            self.logs.add_log("num_round", best_round)
            for server in self.server_list:
                server.global_model.load_state_dict(best_state_dicts[f"{server.farm_name}_best_dict"])
        else:
            self.logs.add_log("num_round", i)
        
        for server in self.server_list:
            for client in server.client_list:
                if self.name != "Local":
                    client.model.load_state_dict(server.global_model.state_dict())
                    client.global_model.load_state_dict(server.global_model.state_dict())
    
