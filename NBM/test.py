import os, pickle, sys, copy
sys.path.insert(0, os.path.join(os.getcwd()))

from typing import Dict, Union
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from datetime import timedelta
from functools import partial
sys.path.append(os.path.join(sys.path[0], r"..\data"))
sys.path.append(os.path.join(os.getcwd(), r"data"))
from Data import data_farm
from time import sleep
import tqdm
sys.path.append(os.path.join(os.getcwd(), r"NBM"))
from exp_utils import PATHClass, make_clients, select_group, round_data, put_position, smoothing, bool_to_txt, cold_start_display, cold_start_display_per, display_output
from experiment_parameter import *
from model_parameter import *
from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm
from multiprocessing import Pool, freeze_support, RLock

import argparse

sys.path.append(os.path.join(sys.path[0], r"..\federated_learning"))
# from utils import MMDLoss
from utils import MMDLoss, RBF

def make_mmd_df(bid, time_range, data, time_windows = 7, time_step = 24, verbose = True): 
    
    
    start_test = pd.to_datetime(time_range[0]) + timedelta(days= 12*7+1 ) 
    end_test = pd.to_datetime(time_range[0]) + timedelta(days= 7*16+1)

    # Kelmarsh
    X = {}
    X_train, timestamps_train = {},{}
    X_test, timestamps_test = {},{}

    for t in exp.kelmarsh_turbine:
        X[t] = data[t]
        filt = (pd.to_datetime(time_range[0]) <= pd.to_datetime(data[t].index)) & (pd.to_datetime(time_range[1]) >= pd.to_datetime(data[t].index))
        X_train[t], timestamps_train[t] = X[t][filt] if X[t] is not None else [], pd.to_datetime(data[t].index)[filt] if pd.to_datetime(data[t].index) is not None else []
       

    current_time = start_test
    MMD = MMDLoss(RBF(device = torch.device("cuda", 0)))
    res = pd.DataFrame(columns=["start", "end", "MMD", "client", "num_test", "num_train"])
    id = 0
    n = 0

    
    tm = copy.deepcopy(current_time)
    while tm < end_test:
        tm = tm + timedelta(minutes= 10*time_step)
        n +=1

    with tqdm(total= n*len(exp.kelmarsh_turbine), desc=f"{time_range[0][:10]} to {time_range[1][:10]}", position=bid+1) as pbar:
        while current_time < end_test:
            for t in exp.kelmarsh_turbine:
                filt = (current_time <=  pd.to_datetime(data[t].index)) & (current_time + timedelta(days = time_windows) >=  pd.to_datetime(data[t].index))
                X_test[t], timestamps_test[t] = X[t][filt] if X[t] is not None else [],  pd.to_datetime(data[t].index)[filt] if  pd.to_datetime(data[t].index)[filt] is not None else []
                
                r = np.mean([MMD(torch.unsqueeze(torch.tensor(X_train[t].values[:,i], device = torch.device("cuda",0)), dim=1),\
                                 torch.unsqueeze(torch.tensor(X_test[t].values[:,i], device = torch.device("cuda",0)), dim=1)).to("cpu") \
                                    for i in range(X_train[t].shape[-1])])
                if not np.isnan(0):
                    res.loc[id] = [str(current_time).split(" ")[0], str(current_time + timedelta(days = time_windows)).split(" ")[0], r, " ".join(t.split("_")[-2:]), \
                                len(X_train), len(X_test)]  
                id +=1
                pbar.update(1)
                

            current_time = current_time + timedelta(minutes= 10*time_step)
            
    res.to_csv(f"{args.path}/{time_range[0][0:10]} to {time_range[1][0:10]}.csv")
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='MMD data creator',
                    description='Create and save MMD data to check for anomalous test sets',
                    epilog='')
    parser.add_argument("-tw", "--time_window", type=int, default = 7, help='Time window considered in the test set in days (default 7)')
    parser.add_argument("-ts", "--time_step", type=int, default = 580, help='Time window considered in the test set in step of 10 minutes (default 24)')
    
    parser.add_argument("-v", "--verbose", action = "store_false", default = 3, help='Verbose (default True)')
    parser.add_argument("--start", type=int, default = 3, help='First time range considered (default = 3)')
    parser.add_argument("--end", type=int, default = -1, help='Last time range considered (default = -1)')
    parser.add_argument("--step", type=int, default = 6, help='Step time range considered (default = 6)')
    
    parser.add_argument("--path", type=str, default = r"NBM/.temp/", help='Path to save')
    args = parser.parse_args()
    experiment_name = "Exp_1"
    file_name = f"result data {experiment_name}"
    
    BASE_PATH = r"C:\Users\goa7\Documents\FL_code\NBM\results"
    path_obj = PATHClass(BASE_PATH=BASE_PATH)
    path_obj.exp_name = experiment_name

    #Loading the corresponding parameters    
    with open(os.path.join(path_obj.EXP_PATH, f"{experiment_name}_exp_parameters.pkl"), "rb") as f:
        exp = pickle.load(f)

        
    with open(os.path.join(path_obj.EXP_PATH, f"{experiment_name}_model_parameters.pkl"), "rb") as f:
        model = pickle.load(f)

 
    data = {}
    for t in exp.kelmarsh_turbine:
        data[t] = pd.read_csv(Path(PurePath(r"C:\Users\goa7\Documents\FL_code\data\data_prep", f"Kelmarsh_{t}.csv")), index_col= "Timestamp", parse_dates= True,\
                              usecols=["Timestamp", "Wind_speed", "Wind_dir_cos", "Wind_dir_sin", "Amb_temp"])
    f = partial(make_mmd_df, data = data, time_windows = args.time_window, time_step = args.time_step, verbose = args.verbose)
    """res = {}

    result_list = {}
    for bid, tr in enumerate(exp.all_dates_range[args.start:args.end:args.step]):
        f(bid, tr)
        torch.cuda.empty_cache()
    """

    
