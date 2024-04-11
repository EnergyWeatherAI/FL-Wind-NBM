import sys, copy, os, pickle

from datetime import timedelta
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt 
import matplotlib.ticker as mticker

sys.path.append(os.path.join(sys.path[0], r"../federated_learning"))
from FederatedAlgorithms import FedAvg, FedProx
from utils import logs, memory_size

sys.path.append(os.path.join(sys.path[0], r"../data"))
from Data import data_farm

import torch
from torchmetrics.regression import MeanAbsoluteError
from model_parameter import model_param
from experiment_parameter import exp_param, BASE_PATH

def federated_training(model, cross_farm, algo, clients, clients_penmanshiel, clients_kelmarsh, clients_EDP, Path, RESET = False, **kwargs): 
    
    Path.cross_farm = cross_farm
    Path.algo = algo
    if not cross_farm and algo == "FedAvg":
        for farm, client_list in zip(["Penmanshiel", "Kelmarsh", "EDP"], [clients_penmanshiel, clients_kelmarsh, clients_EDP]):
            print("\n" + farm + "\n")
            Path.farm = copy.copy(farm)
            if not os.path.isfile(Path.create_full_path(is_model = False, log= False)) or not os.path.isfile(Path.create_full_path(is_model = True, log= False)) or RESET:
                server = FedAvg(model_class=model.model_class, client_list=copy.deepcopy(client_list), init_state_dict = copy.deepcopy(model.model.state_dict()), **model.model_args, **kwargs)
                server.training(evaluation_args = model.evaluation_args, earlystopping = model.earlystopping)
                Path.create_full_path(is_model = False)
                server.logs.save(file_dir=Path.path, file_name=Path.file_name.split(".")[0])
                torch.save(server.global_model.state_dict(), Path.create_full_path(is_model= True))
                del server
                torch.cuda.empty_cache()
        
    elif not cross_farm and algo == "FedProx":

        for farm, client_list in zip(["Penmanshiel", "Kelmarsh", "EDP"], [clients_penmanshiel, clients_kelmarsh, clients_EDP]):
            print("\n" + farm + "\n")
            Path.farm = copy.copy(farm)
            if not os.path.isfile(Path.create_full_path(is_model = False, log= False)) or not os.path.isfile(Path.create_full_path(is_model = True, log= False)) or RESET:
                server = FedProx(model_class=model.model_class, client_list=copy.deepcopy(client_list), init_state_dict = copy.deepcopy(model.model.state_dict()), **model.model_args, **kwargs)
                server.training(evaluation_args = model.evaluation_args, earlystopping = model.earlystopping)
                Path.create_full_path(is_model = False)
                server.logs.save(file_dir=Path.path, file_name=Path.file_name.split(".")[0])
                torch.save(server.global_model.state_dict(), Path.create_full_path(is_model= True))
                del server
                torch.cuda.empty_cache()
        
    elif cross_farm and algo == "FedAvg":
        server = FedAvg(model_class=model.model_class, client_list=copy.deepcopy(clients), init_state_dict = copy.deepcopy(model.model.state_dict()), **model.model_args, **kwargs)
        if not os.path.isfile(Path.create_full_path(is_model = False, log= False)) or not os.path.isfile(Path.create_full_path(is_model = True, log= False)) or RESET:
            server.training(evaluation_args = model.evaluation_args, earlystopping = model.earlystopping)
            Path.create_full_path(is_model = False)
            server.logs.save(file_dir=Path.path, file_name=Path.file_name.split(".")[0])
            torch.save(server.global_model.state_dict(), Path.create_full_path(is_model= True))
            del server
            torch.cuda.empty_cache()

    elif cross_farm and algo == "FedProx":
        server = FedProx(model_class=model.model_class, client_list=copy.deepcopy(clients), init_state_dict = copy.deepcopy(model.model.state_dict()), **model.model_args, **kwargs)
        if not os.path.isfile(Path.create_full_path(is_model = False, log= False)) or not os.path.isfile(Path.create_full_path(is_model = True, log= False)) or RESET:
            server.training(evaluation_args = model.evaluation_args, earlystopping = model.earlystopping)
            Path.create_full_path(is_model = False)
            server.logs.save(file_dir=Path.path, file_name=Path.file_name.split(".")[0])
            torch.save(server.global_model.state_dict(), Path.create_full_path(is_model= True))
            del server
            torch.cuda.empty_cache()

def finetuning(model, cross_farm, algo, client, Path, RESET = False, TEST = False):
    Path.algo = algo
    Path.cross_farm = cross_farm
    Path.client_name = client.name
    LoadPath = PATHClass(Path.BASE_PATH)
    LoadPath.__dict__ = copy.deepcopy(Path.__dict__)
    LoadPath.finetuned = False
    
    farm = client.name.split("_")[0] if not cross_farm else ""
    Path.farm = farm
    LoadPath.farm = farm
    if not os.path.isfile(Path.create_full_path(is_model = False, log= False)) or not os.path.isfile(Path.create_full_path(is_model = True, log= False)) or RESET:
        if os.path.isfile(LoadPath.create_full_path(is_model = True, log = False)) and not TEST:
            
            init_dict = torch.load(LoadPath.create_full_path(is_model = True))
            
            client.prepare_model(model.model_class, state_dict = copy.deepcopy(init_dict), verbose = False, **model.model_args_ft) 
            client.train(epochs = model.epochs_ft, batch_size= model.batch_size_ft, lr = model.lr_ft, device = model.device, log_args = model.log_arg_client(client, model.metric_ft))
            Path.create_full_path(is_model = False)
            client.logs.save(file_dir = Path.path, file_name = Path.file_name.split(".")[0])
            torch.save(client.model.state_dict(),  Path.create_full_path(is_model = True))
            torch.cuda.empty_cache()
        else:
            print(f"MISSING FILE {LoadPath.create_full_path(is_model = True, log = False)}")       

def start_exp(exp, model, RESET = False, SKIP = False, SKIP_LOCAL = False, SKIP_FINETUNING = False, **kwargs):
    # Setup the main folder
    Path = PATHClass(BASE_PATH=BASE_PATH)
    Path.create_base_path(exp_name = exp.name)
    PARAM_PATH = Path.path

    model.reset_model()

    with open(os.path.join(PARAM_PATH, f"{exp.name}_exp_parameters.pkl"), 'wb') as f:
        pickle.dump(exp, f)
    with open(os.path.join(PARAM_PATH, f"{exp.name}_model_parameters.pkl"), 'wb') as f:
        pickle.dump(model, f)

    for time_range in exp.all_dates_range:
        start_date, end_date = time_range[0].split(" ")[0], time_range[1].split(" ")[0] 
        Path.time_range = time_range
        
        # Initialize clients data and the model
        print("\n")
        clients, clients_penmanshiel, clients_kelmarsh, clients_EDP = make_clients(exp, time_range= time_range)  

        print("\n")
        
        
        # First train local model
        if "Local" in exp.all_algorithm:
            Path.algo = "Local"
            for client in clients:
                Path.client_name = client.name 
                if (not (os.path.isfile(Path.create_full_path(is_model = True, log = False)) and os.path.isfile(Path.create_full_path(is_model = False, log = False)))\
                    or RESET) and not SKIP_LOCAL:
                    print(f"Local training of {client.name}")
                    client.prepare_model(model.model_class, state_dict = copy.deepcopy(model.model.state_dict()), verbose = False, **model.model_args) 
                    # client.prepare_data() already done in make_client
                    client.train(epochs = model.epochs_ft, batch_size= model.batch_size_ft, lr = model.lr_ft, device = model.device, \
                                log_args = model.log_arg_client(client, model.metric), verbose = False)
                    Path.create_full_path(is_model= False)
                    client.logs.save(file_dir = Path.path, file_name = Path.file_name.split(".")[0])
                    torch.save(client.model.state_dict(), Path.create_full_path(is_model = True))
                    torch.cuda.empty_cache()
            
            print("\n")

        # Training federated models
        Path.finetuned = False
        for cross_farm in exp.all_cross_farm:
            Path.cross_farm = cross_farm
            cross_text = "cross_farm" if cross_farm else "intra_farm"
            print(f"Training of the {cross_text} algorithms")
            for algo in [name for name in exp.all_algorithm if name != "Local"]:
                Path.algo = algo
                print(f"Training of {algo}")
                print("\n")
                # Start and save the training 
                federated_training(model, cross_farm, algo, clients, clients_penmanshiel, clients_kelmarsh, clients_EDP, Path, RESET=RESET, **kwargs)
                print("\n")
                

        
        # Finetuning
        print("Finetuning")
        Path.finetuned = True
        for cross_farm in exp.all_cross_farm:
            Path.cross_farm = cross_farm
            cross_text = "cross_farm" if cross_farm else "intra_farm"
            print(f"Finetuning of the {cross_text} algorithms")
            for algo in [name for name in exp.all_algorithm if name not in  ["Local"]]:
                Path.algo = algo
                print(f"Finetuning of {algo}")
                                
                for client in clients:
                    Path.client_name = client.name 
                    if (not os.path.isfile(Path.create_full_path(is_model = True, log = False)) or os.path.isfile(Path.create_full_path(is_model = False, log = False))\
                        or RESET) and not SKIP_FINETUNING:
                        print(f"Finetuning of {client.name} for {algo} with {cross_text}")
                        finetuning(model, cross_farm, algo, client, Path, RESET = RESET)
        
def eval_model(model, Path, client, metric):
    Path.client_name = client.name
    Path.farm = client.name.split("_")[0]
    if os.path.isfile(Path.create_full_path(is_model = True)):
        model.model.load_state_dict(torch.load(Path.full_path))
        device = model.device
        X,y = client._X_test, client._y_test
        if "to" in dir(metric):
            metric = metric.to(device)
        else:
            metric = metric
        r = metric(model.model.to(device)(X.to(device)), y.to(device))
        return r.to("cpu").detach().numpy(), memory_size(model.model.state_dict())
    else:
        print(Path.full_path)
        return np.nan, np.nan

def time_diff(res):
    res["time_diff"] = res["end"].apply(pd.to_datetime) - res["start"].apply(pd.to_datetime) 
    return res

def add_farm(res):
    res["farm"] = res["client"].apply(lambda x : x.split("_")[0])
    return res      
        
def create_results_dataframe(exp = exp_param(), model = model_param(), file_name = "results", verbose = True, \
                             exp_name = "Exp_1", BASE_PATH = "results", SAVE_PATH = None, metric = MeanAbsoluteError()):
    id = 0
    Path = PATHClass(BASE_PATH=BASE_PATH)
    Path.exp_name = exp_name
    
    res = pd.DataFrame(columns=["start", "end", "client", "cross_farm", "algo", "finetuned", "metric", "nb_train", "nb_test", "res", "num_round",\
                                "model_size", "data_train_weight",  "data_test_weight", "exp_name"])
    
    # Model inititialization
    model.reset_model()
    
    for time_range in Path.get_time_range_list():
        start_date, end_date = time_range[0].split(" ")[0], time_range[1].split(" ")[0]
        Path.time_range = time_range 
        print("\n")
        clients, clients_penmanshiel, clients_kelmarsh, clients_EDP = make_clients(exp, time_range= time_range)  

        
        for algo, cross_farm in Path.get_algo_and_cross_farm():
            if cross_farm is None:
                cross_text = ""
            else:
                cross_text = "cross farm, " if cross_farm else "intra farm, "
            Path.algo = algo
            Path.cross_farm = cross_farm
            for client in clients:
                for finetuned in [False, True] if algo != "Local" else [True]:
                    finetuned_text = "and finetuned" if finetuned else ""
                    Path.finetuned = finetuned

                     
                    print(f"Evaluation with time range from {start_date} to {end_date}, metric {metric}, algo {algo}, {cross_text} client {client.name} {finetuned_text}")
                    r, size = eval_model(model, Path, client, metric)
                    turbine = "_".join(client.name.split("_")[1:])
                    Path.farm = client.name.split("_")[0]
                    normal = {}
                    with open(f"../data/data_normalisation/std_mean_normalisation_{Path.farm}.pkl", "rb") as f:
                        normal = pickle.load(f)
                    std = normal[turbine][exp.cols[0]][1]
                    r = r*std

                    # Get number of communication to best_round
                    if os.path.isfile(Path.create_full_path(is_model = False)):
                        with open( Path.create_full_path(is_model = False), "rb") as f:
                            log = pickle.load(f)
                        num_round = log["num_round"] if 'num_round' in log.keys() else np.nan
                    else:
                        num_round = np.nan

                    # Write the row
                    res.loc[id] = [time_range[0],time_range[1],client.name, cross_farm if cross_farm is not None else "NA", algo, finetuned, \
                            metric.__class__.__name__, len(client._X_train), len(client._X_test), r, num_round, size,\
                                memory_size(client._X_train) + memory_size(client._y_train), \
                                memory_size(client._X_test) + memory_size(client._y_test), exp_name]
                    id+=1
                    if verbose:
                        print(f"Result: {r}")
                        
   
    res = time_diff(res)
    res = add_farm(res)
    res.to_pickle(os.path.join(Path.EXP_PATH, f"{file_name}.pkl") if SAVE_PATH is None else SAVE_PATH)    
    return res       

def make_clients(exp, time_range = "all", **args): 
    
    
    col_dict_in = {'Wind_speed':0, 'Wind_dir_cos':1, 'Wind_dir_sin':2, 'Amb_temp':3}
    col_dict_out = {'Power':0, 'Gear_bear_temp_avg':1, 'Rotor_speed':2, 'Gen_speed':3, \
                    'Gen_bear_temp_avg':4,'Nacelle_position_cos':5, 'Nacelle_position_sin':6}

    print(f"Preparing clients from {time_range[0]} to {time_range[1]}\n")

    turbine_names_penmanshiel = ["WT_01", "WT_02", "WT_15", "WT_04", "WT_05", "WT_06", "WT_07", "WT_08", "WT_09", "WT_10", "WT_11", "WT_12", "WT_13", "WT_14"]
    DATA_PATH_PENMANSHIEL = r'..\data\Penmanshiel'

    farm_penmanshiel = data_farm(cols = exp.cols, name = "Penmanshiel", list_turbine_name= turbine_names_penmanshiel, DATA_PATH=DATA_PATH_PENMANSHIEL,\
                                 initialize = False, PATH= r"..\data\data_prep")


    turbine_names_kelmarsh = ["WT_01", "WT_02", "WT_03", "WT_04", "WT_05", "WT_06"]
    DATA_PATH_KELMARSH = r'..\data\Kelmarsh'

    farm_kelmarsh = data_farm(cols = exp.cols, name = "Kelmarsh", list_turbine_name= turbine_names_kelmarsh, DATA_PATH=DATA_PATH_KELMARSH,\
                               initialize = False, PATH= r"..\data\data_prep")


    turbine_names_EDP= ["WT_01", "WT_06", "WT_07", "WT_11"]
    DATA_PATH_EDP = r'..\data\EDP'

    farm_EDP = data_farm(cols = exp.cols, name = "EDP", list_turbine_name= turbine_names_EDP, DATA_PATH=DATA_PATH_EDP, initialize = False,\
                          PATH= r"..\data\data_prep"
                         )

    client_list = []
    client_penmanshiel = []
    client_kelmarsh = []
    client_EDP = []

    col_nums = [col_dict_out[key] for key in exp.cols] if exp.cols != "all" else "all"
    client_penmanshiel = farm_penmanshiel.make_clients(turbine_names = exp.penmanshiel_turbine, time_range = time_range, cols_num= col_nums,\
                                                       data_threshold = exp.data_threshold,**args)
    print("__________________________________________________________")
    client_kelmarsh =farm_kelmarsh.make_clients(turbine_names = exp.kelmarsh_turbine, time_range = time_range, cols_num= col_nums,\
                                                 data_threshold = exp.data_threshold, **args)
    print("__________________________________________________________")
    client_EDP = farm_EDP.make_clients(turbine_names = exp.EDP_turbine, time_range = time_range, cols_num= col_nums, \
                                       data_threshold = exp.data_threshold, **args)
    print("__________________________________________________________")

    # Prepare the test set
    start_test = pd.to_datetime(time_range[0]) + timedelta(days= 7*exp.max_week +1) 
    end_test = start_test + timedelta(days= 7*exp.nb_week_test)
    
    # Penmanshiel
    X,y,timestamps = {},{},{}
    for t, client in [(c.name, c) for c in client_penmanshiel]:
        X[t], y[t], timestamps[t] = farm_penmanshiel.get_data_window("_".join(t.split("_")[-2:]))
        if col_nums != "all":
            y[t] = y[t].squeeze(dim= 1)[:, col_nums]

        filt = (start_test <= pd.to_datetime(timestamps[t])) & (end_test >= pd.to_datetime(timestamps[t]))
                    
        X[t], y[t], timestamps[t] = X[t][filt] if X[t] is not None else [], y[t][filt] if y[t] is not None else [], \
        timestamps[t][filt] if timestamps[t] is not None else []

        client._X_test = X[t]
        client._y_test = y[t]
        client.timestamps_test = timestamps[t]
    
    # Kelmarsh
    X,y,timestamps = {},{},{}
    for t, client in [(c.name, c) for c in client_kelmarsh]:
        X[t], y[t], timestamps[t] = farm_kelmarsh.get_data_window("_".join(t.split("_")[-2:]))
        if col_nums != "all":
            y[t] = y[t].squeeze(dim= 1)[:, col_nums]
        

        filt = (start_test <= pd.to_datetime(timestamps[t])) & (end_test >= pd.to_datetime(timestamps[t]))
                    
        X[t], y[t], timestamps[t] = X[t][filt] if X[t] is not None else [], y[t][filt] if y[t] is not None else [], \
        timestamps[t][filt] if timestamps[t] is not None else []

        client._X_test = X[t]
        client._y_test = y[t]
        client.timestamps_test = timestamps[t]

    # EDP
    X,y,timestamps = {},{},{}
    for t, client in [(c.name, c) for c in client_EDP]:
        X[t], y[t], timestamps[t] = farm_EDP.get_data_window("_".join(t.split("_")[-2:]))
        if col_nums != "all":
            y[t] = y[t].squeeze(dim= 1)[:, col_nums]

        filt = (start_test <= pd.to_datetime(timestamps[t])) & (end_test >= pd.to_datetime(timestamps[t]))
                    
        X[t], y[t], timestamps[t] = X[t][filt] if X[t] is not None else [], y[t][filt] if y[t] is not None else [], \
        timestamps[t][filt] if timestamps[t] is not None else []

        client._X_test = X[t]
        client._y_test = y[t]
        client.timestamps_test = timestamps[t]

        client_list = client_penmanshiel + client_kelmarsh + client_EDP
    
    return client_list, client_penmanshiel, client_kelmarsh, client_EDP

def get_selection(df, condition):
    df = copy.deepcopy(df)
    col = df.columns
    for key, value in condition.items():
        df[f"{key}_cond"] = df[key] == value if not isinstance(value, list) else df[key].apply(lambda x : x in value)
    filt = df[[f"{key}_cond" for key in condition.keys()]].all(axis = "columns") 
    return df[col].loc[filt]

def select_group(df, target, group_by, condition):
    new_res = copy.deepcopy(get_selection(df, condition))
    new_res = new_res[target+group_by].groupby(by = group_by).mean()
    return new_res

def round_number(n, depth = 2):
    if isinstance(n, str):
        return n
    else:
        return np.round(n, depth)
    
def round_data(df, target = "res", depth = 2):
    for i in df.index:
        df.loc[i][target] = round_number(df.loc[i][target], depth)
    return df

def put_position(df, col, pos =0):
    # shift column 'Name' to first position 
    first_column = df.pop(col) 
    df.insert(pos, col, first_column) 
    return df

def smoothing(df, freq = "10min", kind = "quadratic"):


    df = copy.deepcopy(df)
    time_diff = df.index
    fake_start = pd.to_datetime("2016-06-01")
    df.index = [fake_start + dt for dt in df.index]
    df2 = df.asfreq(freq)
    new_index = df2.index
    #df2 = df2.interpolate(method = method)
    for c in df.columns:
        f = interp1d(pd.to_numeric(df.index), df[c],kind= kind)
        df2[c] = f(pd.to_numeric(new_index))
    df2.index = [time_diff[0]] + [time_diff[0] + new_index [i+1] - new_index [0] for i in range(len(new_index)-1)]   

    return df2

def smoothing_exp(df, freq = "10min", kind = "quadratic"):


    df = copy.deepcopy(df)
    time_diff = df.index
    fake_start = pd.to_datetime("2016-06-01")
    df.index = [fake_start + dt for dt in df.index]
    df2 = df.asfreq(freq)
    new_index = df2.index
    #df2 = df2.interpolate(method = method)
    for c in df.columns:
        df2[c] = df2[c].apply(np.log)
        f = interp1d(pd.to_numeric(df.index), df[c].apply(np.log),kind= kind)
        df2[c] = f(pd.to_numeric(new_index))
        df2[c] = df2[c].apply(np.exp)
    df2.index = [time_diff[0]] + [time_diff[0] + new_index [i+1] - new_index [0] for i in range(len(new_index)-1)]   

    return df2

def bool_to_txt(x ,txt_true = "Cross-farm", txt_false = "Intra-farm", na = "NA"):
    if x == "NA" or x == np.nan:
        return na
    elif x:
        return txt_true
    elif not x:
        return txt_false
    else:
        return "ERROR"

def cold_start_display(df, algo, colormap, axs = None, conditions = {"cross_farm": [False, "NA"], 'algo':["Local", "FedAvg"], "finetuned": True},\
             title = None, inter_position = (120,160), text_position = (12, 2.5), fig = None, relative = 'offset points', kind = "cubic"):
    # Input 
    conditions["finetuned"] = True
    
    temp_res = select_group(df = df, target=["res"], group_by=["time_diff", "algo"], condition=conditions)
    temp_res = round_data(temp_res)
    temp_res = temp_res.unstack(1, fill_value= np.nan)
    temp_res = put_position(temp_res, ("res", 'Local')) 
    temp_res = smoothing(temp_res, kind = kind)
    
    best_local = temp_res[("res", "Local")].min()
    best_local_time = temp_res[("res", "Local")].argmin()
    
    # x_intersect
    
    
    index = temp_res.index
    fig, ax = plt.subplots() if axs is None else fig, axs

    temp_res.plot(legend = True, ax = ax, cmap = colormap)
    #ax.vlines(x=[x_intersect], ymin=0, ymax=best_local, colors='red', ls='--')
    
    x_intersect = temp_res[("res", algo)][temp_res[("res", algo)]<best_local].index[0] if len(temp_res[("res", algo)][temp_res[("res", algo)]>best_local].index) > 0\
          else temp_res.index[0]
    x_ns = x_intersect / pd.Timedelta(1,'ns')
    
    ax.plot(index, [best_local if t >= x_intersect else np.nan for  t in index], color = "red", linestyle = "--", linewidth=1) 
    ax.annotate(str(x_intersect).split(" ")[:1][0], xy=(x_ns, best_local), xytext=inter_position, textcoords=relative,\
                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1, linestyle = "--"))
    
    ax.plot(x_ns, best_local, 'ro')
    ax.vlines(x=[x_ns], ymin = 0, ymax = best_local, color='r', linestyle='--', linewidth=1)
    ax.xaxis.set_tick_params(rotation=45)

    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ticks_label = ax.get_xticklabels()
    l = [" ".join(x.get_text().split(" ")[:-1]) for x in ticks_label]
    ax.set_xticklabels(l)
    
    ax.set_ylabel('MAE')
    titl = "MAE cold start for FedAvg" if title is None else title
    ax.set_title(titl)
    ax.set_ylim(bottom=0)
    ax.legend([col[1] for col in temp_res.columns.unique()])  
    td =  temp_res.index[best_local_time] - x_intersect 
    td = td.round(freq= "d")
    td = " ".join(str(td).split(" ")[:-1])
    
    ax.text(text_position[0]*24*3600*1000000000, text_position[1], f"Speed up by {td}")
    fig.supxlabel("Time range")
    return td

def cold_start_display_per(df, algo, colormap, axs = None, conditions = {"cross_farm": [False, "NA"], 'algo':["Local", "FedAvg"], "finetuned": True},\
             title = None, inter_position = (120,160), loc_pos= (-40,40), text_position = (12, 2.5), fig = None, relative = 'offset points',\
                  kind = "cubic", tol_perc = 0, loc_relative = 'axes fraction'):
    # Input 
    conditions["finetuned"] = True
    
    temp_res = select_group(df = df, target=["res"], group_by=["time_diff", "algo"], condition=conditions)
    temp_res = round_data(temp_res)
    temp_res = temp_res.unstack(1, fill_value= np.nan)
    temp_res = put_position(temp_res, ("res", 'Local')) 
    temp_res = smoothing(temp_res, kind = kind)
    
    best_local = temp_res[("res", "Local")].min()
    best_local_time = temp_res[("res", "Local")].idxmin()
    fig, ax = plt.subplots() if axs is None else fig, axs

    assert tol_perc >= 0, "tol_perc argument must be positive or 0"
    tol_factor = 1. + tol_perc/100
    tol_best_local_time  = temp_res[("res", "Local")][temp_res[("res", "Local")]<=best_local*tol_factor].index[0] 
    tol_best_local = temp_res[("res", "Local")][tol_best_local_time]
    
    # x_intersect
    
    
    index = temp_res.index
    temp_res.plot(legend = True, ax = ax, cmap = colormap)
    #ax.vlines(x=[x_intersect], ymin=0, ymax=best_local, colors='red', ls='--')
    
    x_intersect = temp_res[("res", algo)][temp_res[("res", algo)]>best_local*tol_factor].index[-1] if len(temp_res[("res", algo)][temp_res[("res", algo)]>best_local*tol_factor].index) > 0\
          else temp_res.index[0]
    y_intersect = temp_res[("res", algo)][x_intersect] if x_intersect != index[0] else tol_best_local
    x_ns = x_intersect / pd.Timedelta(1,'ns') 
    #ax.plot([0,tol_best_local_time/ pd.Timedelta(1,'ns')], [0, temp_res[("res", "Local")][tol_best_local_time]], color = "red", linestyle = "--", linewidth=1)
    ax.plot([x_ns, tol_best_local_time/ pd.Timedelta(1,'ns')], [y_intersect, tol_best_local], color = "red", linestyle = "--", linewidth=1)
    #ax.plot(index, [tol_best_local if t >= x_intersect and t <= tol_best_local_time else np.nan for  t in index], color = "red", linestyle = "--", linewidth=1)
    ax.annotate(str(x_intersect).split(" ")[0] + " days", xy=(x_ns, y_intersect), xytext=inter_position, textcoords=relative,\
                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1, linestyle = "--"))
    ax.plot(x_ns, y_intersect, 'ro')

    ax.annotate(str(tol_best_local_time).split(" ")[0] + " days", xy=(tol_best_local_time/ pd.Timedelta(1,'ns'), tol_best_local), xytext=loc_pos,\
                 textcoords=loc_relative,\
                 arrowprops=dict(arrowstyle='->', color='black', linewidth=1, linestyle = "--"))
    ax.plot(tol_best_local_time/ pd.Timedelta(1,'ns'), tol_best_local, 'ro')
    ax.vlines(x=[x_ns], ymin = 0, ymax = y_intersect, color='r', linestyle='--', linewidth=1)
    ax.xaxis.set_tick_params(rotation=45)

    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ticks_label = ax.get_xticklabels()
    l = [" ".join(x.get_text().split(" ")[:-1]) for x in ticks_label]
    ax.set_xticklabels(l)
    ax.set_ylabel('MAE')
    titl = "MAE cold start for FedAvg" if title is None else title
    ax.set_title(titl)
    ax.set_ylim(bottom=0)
    ax.legend([col[1] for col in temp_res.columns.unique()])  
    td =  tol_best_local_time - x_intersect 
    td = td.round(freq= "d")
    td = " ".join(str(td).split(" ")[:-1])
    
    ax.text(text_position[0], text_position[1], f"Speed up by {td}", transform=ax.transAxes)
    fig.supxlabel("Time range")
    return td

def fill_na(time_range, exp, y, timestamp, window = "all", start = 0, stride = 1):
    
    start_test = pd.to_datetime(time_range[0]) + timedelta(days= 7*exp.max_week +1) 
    end_test = start_test + timedelta(days= 7*exp.nb_week_test)

    full = pd.date_range(start=start_test, end=end_test, freq="10min")
    for i, t in enumerate(full):
        if t not in timestamp:
            y, timestamp = np.insert(y, i,  np.nan), np.insert(timestamp, i, t)
    if window != "all":
        y, full = y[start*6*24: start*6*24+ window*6*24 : stride], full[start*6*24: start*6*24+ window*6*24 : stride]
    
    if all([np.isnan(x) for x in y]):
        return np.array([np.nan for _ in range(len(np.arange(start*6*24, start*6*24+ window*6*24, stride)))]), full
    max_id = np.max([id for id, y in enumerate(y) if not np.isnan(y)])
    min_id = np.min([id for id, y in enumerate(y) if not np.isnan(y)])

    return np.array(y[min_id: max_id+1:stride]), full[min_id:max_id+1:stride]

def display_output(time_range, exp, model, Path, clients, window = "all", start = 0, stride =1, farm = None,\
                    adjust_dict = {"top" : 0.9, "bottom" : 0.2, "wspace": 0.125}, local_line = 1, line = 0.5, local_alpha = 1, alpha = 0.5):
    
    if farm == None: 
        fig, axs = plt.subplots(3,3, sharex = "col", sharey = True, figsize = (18,10))
        
        # One plot per farm each display ground truth, intra farm, cross farm and Local (picking turbine WT_01)
        fig.subplots_adjust(**adjust_dict)
        Path.time_range = time_range
        
        for i, farm in enumerate(exp.list_farm_names):
            # Test time
            Path.farm = farm
            client = [c for c in clients if farm in c.name][0] 
            turbine = " ".join(client.name.split("_")[-2:])
            

            if window == "all":
                X, y_truth, timestamp = client._X_test[::stride], client._y_test[::stride], client.timestamps_test[::stride]
            else:
                X, y_truth, timestamp = client._X_test[start*6*24:start*6*24 + window*6*24 :stride], client._y_test[start*6*24:start*6*24 + window*6*24 :stride], client.timestamps_test[start*6*24:start*6*24 + window*6*24 :stride]

            with open(f"../data/data_normalisation/std_mean_normalisation_{Path.farm}.pkl", "rb") as f:
                normal = pickle.load(f)
            std = normal["_".join(client.name.split("_")[1:])][exp.cols[0]][1]
            mean = normal["_".join(client.name.split("_")[1:])][exp.cols[0]][0]

            

            # Ground truth
            for j in range(3):
                axs[j][i].xaxis.set_tick_params(rotation=45)  
                y_t, ts = fill_na(time_range, exp, copy.deepcopy(y_truth), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
                y_t = y_t*std + np.ones_like(y_t)*mean
                    
                ticks_loc = axs[j][i].get_xticks().tolist()
                axs[j][i].plot(pd.to_datetime(ts), y_t, label = "Ground truth", color = "black", linewidth = local_line, alpha = local_alpha)
                axs[0][i].set_title(f"{farm} {turbine}")
                

            # FedAvg 
            Path.algo = "FedAvg"
            for j,finetuned in enumerate([False, True]):
                for cross_farm in [False, True]:
                    Path.cross_farm = cross_farm
                    Path.finetuned = finetuned
                    Path.client_name = client.name
                    txt = bool_to_txt(finetuned, txt_true="finetuned", txt_false="not finetuned")
                    if os.path.isfile(Path.create_full_path(is_model = True)):
                        model.model.load_state_dict(torch.load(Path.create_full_path(is_model = True)))
                        y_pred = model.model(X).detach().numpy()
                        y_p, ts = fill_na(time_range, exp, copy.deepcopy(y_pred), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
                        y_p = y_p*std + + np.ones_like(y_p)*mean
                        axs[j+1][i].plot(ts, y_p, label = f"{bool_to_txt(cross_farm)} {txt} FedAvg", color = "red" if cross_farm else "blue", linestyle = "--", linewidth = line, alpha = alpha)
                        axs[j+1][i].legend()
            # Local 
            Path.algo = "Local"
            Path.client_name = client.name
            if os.path.isfile(Path.create_full_path(is_model = True)):
                model.model.load_state_dict(torch.load(Path.create_full_path(is_model = True)))
                y_pred = model.model(X).detach().numpy()
                y_p, ts = fill_na(time_range, exp, copy.deepcopy(y_pred), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
                y_p = y_p*std + + np.ones_like(y_p)*mean
                axs[0][i].plot(ts, y_p, label = f"Local", color = "green", linestyle = "--", linewidth = line, alpha = alpha)        
                axs[0][i].legend()

        for j in range(3):
            ticks_loc = axs[j][0].get_yticks().tolist()
            axs[j][0].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ticks_label = axs[j][0].get_yticklabels()
            #l = [str(int(float(x.get_text()))) for x in ticks_label]
            axs[j][0].set_yticklabels(ticks_label)
            
        start_txt, end_txt = time_range[0].split(" ")[0], time_range[1].split(" ")[0]
        
        fig.suptitle(f"Temperature prediction for time range {start_txt} to {end_txt} for {farm}") 
        fig.supylabel("MAE in °C")
        fig.supxlabel("Test time")
    else: 
        fig, axs = plt.subplots(1,3, sharex = "col", figsize = (18,10))
        fig.subplots_adjust(**adjust_dict)
        Path.time_range = time_range
        Path.farm = farm
        client = [c for c in clients if farm in c.name][0]

        if window == "all":
            X, y_truth, timestamp = client._X_test[::stride], client._y_test[::stride], client.timestamps_test[::stride]
        else:
            X, y_truth, timestamp = client._X_test[start*6*24:start*6*24 + window*6*24 :stride], client._y_test[start*6*24:start*6*24 + window*6*24 :stride], client.timestamps_test[start*6*24:start*6*24 + window*6*24 :stride]

        with open(f"../data/data_normalisation/std_mean_normalisation_{Path.farm}.pkl", "rb") as f:
            normal = pickle.load(f)
        std = normal["_".join(client.name.split("_")[1:])][exp.cols[0]][1]
        mean = normal["_".join(client.name.split("_")[1:])][exp.cols[0]][0]

        # Ground truth
        for j in range(3):
            axs[j].xaxis.set_tick_params(rotation=45)  
            y_t, ts = fill_na(time_range, exp, copy.deepcopy(y_truth), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
            y_t = y_t*std + np.ones_like(y_t)*mean
            axs[j].plot(ts, y_t, label = "Ground truth", color = "black", linewidth = 1)
            ticks_loc = axs[j].get_yticks().tolist()
            axs[j].yaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            ticks_label = axs[j].get_yticklabels()
            l = [str(int(float(x.get_text()))) for x in ticks_label]
            axs[j].set_yticklabels(l)   

        # FedAvg 
        Path.algo = "FedAvg"
        for j,finetuned in enumerate([False, True]):
            for cross_farm in [False, True]:
                Path.cross_farm = cross_farm
                Path.finetuned = finetuned
                Path.client_name = client.name
                txt = bool_to_txt(finetuned, txt_true="finetuned", txt_false="not finetuned")
                if os.path.isfile(Path.create_full_path(is_model = True)):
                    model.model.load_state_dict(torch.load(Path.create_full_path(is_model = True)))
                    y_pred = model.model(X).detach().numpy()
                    y_p, ts = fill_na(time_range, exp, copy.deepcopy(y_pred), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
                    y_p = y_p*std + + np.ones_like(y_p)*mean
                    axs[j+1].plot(ts, y_p, label = f"{bool_to_txt(cross_farm)} {txt} FedAvg", color = "red" if cross_farm else "blue", linestyle = "--", linewidth = 0.5)
                    axs[j+1].legend()
        # Local 
        Path.algo = "Local"
        Path.client_name = client.name
        if os.path.isfile(Path.create_full_path(is_model = True)):
            model.model.load_state_dict(torch.load(Path.create_full_path(is_model = True)))
            y_pred = model.model(X).detach().numpy()
            y_p, ts = fill_na(time_range, exp, copy.deepcopy(y_pred), copy.deepcopy(timestamp), window = window, start = start, stride = stride)
            y_p = y_p*std + + np.ones_like(y_p)*mean
            axs[0].plot(ts, y_p, label = f"Local", color = "green", linestyle = "--", linewidth = 0.5)        
            axs[0].legend()
        
    start_txt, end_txt = time_range[0].split(" ")[0], time_range[1].split(" ")[0]

    fig.suptitle(f"Temperature prediction for time range {start_txt} to {end_txt} for {farm}") 
    fig.supylabel("MAE in °C")
    fig.supxlabel("Test time")
    return fig, axs


class PATHClass:
    def __init__(self, BASE_PATH) -> None:
        self.BASE_PATH = BASE_PATH
        self.exp_name = None
        self.time_range = None
        self.algo = None
        self.cross_farm = None
        self.farm = None
        self.finetuned = None
        self.client_name = None
        self.is_model = None

        self.path = self.BASE_PATH
        self.file_name = None
        self.full_path = None
        self.log = []
    
    def check_create_path(self, path):
        if not os.path.exists(path): 
            os.makedirs(path)

    def create_base_path(self, log = True, **args):
        for key in ["exp_name", "time_range", "algo", "cross_farm"]:
            if key in args.keys():
                setattr(self, key, args[key])

        self.path = self.BASE_PATH
        if self.exp_name is not None:
            new_path = os.path.join(self.path, self.exp_name)
            self.check_create_path(new_path)
            self.path = new_path
        else:
            if log:
                self.log.append((self.path, self.file_name, self.full_path))
            return self.path

        if self.time_range is not None:
            start_date, end_date = self.time_range[0].split(" ")[0], self.time_range[1].split(" ")[0] 
            new_path = os.path.join(self.path, f"{start_date} to {end_date}")
            self.check_create_path(new_path)
            self.path = new_path
        else:
            if log:
                self.log.append((self.path, self.file_name, self.full_path))
            return self.path
        
        if self.algo is not None:
            if self.algo in ["Local", "FarmMixing"]:
                new_path = os.path.join(self.path, self.algo)
                self.check_create_path(new_path)
                self.path = new_path
            else:
                if self.cross_farm is not None:
                    cross_text = "cross_farm" if self.cross_farm else "intra_farm"
                    new_path = os.path.join(self.path, f"{self.algo}_{cross_text}")
                    self.check_create_path(new_path)
                    self.path = new_path
                else:
                    if log:
                        self.log.append((self.path, self.file_name, self.full_path))
                    return self.path
            if log:
                self.log.append((self.path, self.file_name, self.full_path))
            return self.path
        
    def create_file_name(self, log = True, **args): 
        for key in ["algo", "finetuned", "client_name", "is_model", "farm"]:
            if key in args.keys():
                setattr(self, key, args[key])
        assert self.is_model is not None, print("Need the type of file to consider")
        return_type = "model" if self.is_model else "training_log"
        extension = "pth" if self.is_model else "pkl"
        if self.algo == "Local":
            assert self.client_name is not None, print("Need a client name to get the file path for Local")
            self.file_name = f"{self.algo}_{return_type}_{self.client_name}.{extension}"
            if log:
                self.log.append((self.path, self.file_name, self.full_path))
            return self.file_name
        elif self.algo == "FarmMixing":
            assert self.finetuned is not None, print("Need the finetuned parameter")
            assert self.farm is not None, print("Need the farm name")
            if self.finetuned:
                assert self.client_name is not None, print("Need a client name for finetuned results")
                self.file_name = f"{self.algo}_{return_type}_{self.client_name}_finetuned.{extension}"
                if log:
                    self.log.append((self.path, self.file_name, self.full_path))
                return self.file_name
            else:
                self.file_name = f"{self.algo}_{return_type}_{self.farm}.{extension}"
                if log:
                    self.log.append((self.path, self.file_name, self.full_path))
                return self.file_name
        else: 
            assert self.finetuned is not None, print("Need to have the finetuned parameter")
            if self.finetuned:
                assert self.client_name is not None, print("Need a client name for finetuned results")
                self.file_name = f"{self.algo}_{return_type}_{self.client_name}_finetuned.{extension}"
                if log:
                    self.log.append((self.path, self.file_name, self.full_path))
                return self.file_name
            else:
                assert self.cross_farm is not None, print("Need to know is it is cross or intra farm")
                if self.cross_farm:
                    self.file_name =  f"{self.algo}_global_{return_type}.{extension}"
                    self.farm = None
                else:
                    assert self.farm is not None, print("Need to know which farm we are using")
                    self.file_name =  f"{self.algo}_{self.farm}_global_{return_type}.{extension}"
                if log:
                    self.log.append((self.path, self.file_name, self.full_path))
                return self.file_name
     
    def create_full_path(self, log = True, **args):
        for key in ["algo", "finetuned", "client_name", "exp_name", "time_range", "cross_farm", "is_model", "farm"]:
            if key in args.keys():
                setattr(self, key, args[key])
        self.create_base_path(log = False)
        self.create_file_name(log = False)
        self.full_path = os.path.join(self.path, self.file_name)
        
        if log:
            self.log.append((self.path, self.file_name, self.full_path))
        return self.full_path
    
    @property
    def EXP_PATH(self):
        assert self.exp_name is not None, print("Need the experiment name for the EXP_PATH") 
        return os.path.join(self.BASE_PATH, self.exp_name)

    @property
    def TIME_PATH(self):
        assert self.exp_name is not None, print("Need an experiment name")
        assert self.time_range is not None, print("Need a time range")

        # Make the path of interest
        start_date, end_date = self.time_range[0].split(" ")[0], self.time_range[1].split(" ")[0] 
        return os.path.join(os.path.join(self.BASE_PATH, self.exp_name), f"{start_date} to {end_date}")

    @staticmethod
    def dir_to_timerange(dir):
        start, end = dir.split(" ")[0], dir.split(" ")[2]
        start = start + " 18:30:00"
        end = end + " 18:30:00"
        return (start, end)

    @staticmethod
    def dir_to_algo_and_cross_farm(dir):
        all_words = dir.split("_")
        algo = all_words[0]
        if "cross" in all_words:
            cross_farm = True
        elif "intra" in all_words:
            cross_farm = False
        else:
            cross_farm = None 
        return algo, cross_farm

    def get_time_range_list(self, **args):  #Maybe later add other filters
        if "exp_name" in args.keys():
            self.exp_name = args["exp_name"]
        PATH = self.EXP_PATH
        list_dir = [dir for dir in os.listdir(PATH) if os.path.isdir(os.path.join(PATH,dir))]
        list_time_range = list(map(self.dir_to_timerange, list_dir))
        return list_time_range
    
    def get_algo_and_cross_farm(self, **args):
        for key in ["exp_name", "time_range"]:
            if key in args.keys():
                setattr(self, key, args[key])

        PATH = self.TIME_PATH      
        # Check the available directories
        list_dir = [dir for dir in os.listdir(PATH) if os.path.isdir(os.path.join(PATH,dir))]

        # Get infos from the name
        return list(map(self.dir_to_algo_and_cross_farm, list_dir))
    
