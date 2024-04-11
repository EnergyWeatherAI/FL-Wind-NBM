import sys, os
import pickle
from time import time

import pandas as pd
import numpy as np
import torch

sys.path.append(r"../federated_learning")
sys.path.append(os.path.join(os.getcwd(), r"federated_learning"))
from Client import Client

class data_farm:
    def __init__(self, cols, name,  list_turbine_name, DATA_PATH, **kwargs):
        self.name = name
        self.columns = cols
        self.list_turbine_name = list_turbine_name
        self.DATA_PATH = DATA_PATH
        
        default_param = {"in_shift" : 24*6, "out_shift": 1, "in_cols":["Wind_speed", "Wind_dir_cos", "Wind_dir_sin", "Amb_temp"], \
                         "out_cols" : ["Power", "Gear_bear_temp_avg", "Rotor_speed", "Gen_speed","Gen_bear_temp_avg", "Nacelle_position_cos", "Nacelle_position_sin"], \
                         "PATH" : r"data_prep", "RESET" : False, "load_data_window" : False, "initialize": True,\
                         "time_name": "Timestamp", "rename" : {}, "max_workers" : 10
                         }
        
        for key in default_param.keys():
            if key not in kwargs.keys():
                kwargs[key] = default_param[key]
        
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        
        self.normalized_dict = {}
        self.normalized_circ_dict = {}

        # Initializing or loading the turbine data
        if self.initialize:
            for t in self.list_turbine_name:
                print(f"Initializing {t}")
                self.initialize_data(t)
            self.prepare_data()
        else:
            self.load_data()
        
        self.X, self.y, self.timestamps = {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}, {t : None for t in self.list_turbine_name}

    def initialize_data(self, t):
        setattr(self, t, self.load_df(f"{t}.csv", rename = self.rename))
        assert "Timestamp" in getattr(self, t).columns, print("Need a timestamp column")
        assert "Wind_speed" in getattr(self, t).columns, print("Need a Wind_speed column")
        index = pd.to_datetime(getattr(self,t)["Timestamp"]).dt.tz_convert(None) 
        getattr(self,t).set_index(index, inplace = True)
        getattr(self,t).drop(["Timestamp"], axis = 1, inplace= True)
     
    def save_data(self,turbine = None):
        turbine = turbine if turbine is not None else self.list_turbine_name
        
        for t in turbine:
            getattr(self, t).to_csv(os.path.join(self.PATH, f"{self.name}_{t}.csv")) 
    
    def load_data(self, t = None):
        turbine = t if t is not None else self.list_turbine_name
        for t in turbine:
            f =os.path.join(self.PATH, f"{self.name}_{t}.csv")
            assert os.path.isfile(os.path.join(self.PATH, f"{self.name}_{t}.csv")), print(f"File {f} does not exists")
            setattr(self, t, pd.read_csv(os.path.join(self.PATH, f"{self.name}_{t}.csv"), index_col= "Timestamp", parse_dates= True))

    def load_df(self, file_name, rename = {}):
        
        wt_df = pd.read_csv(os.path.join(self.DATA_PATH, file_name), parse_dates=[self.time_name])

        # data cleaning, renaming
        if len(self.rename) > 0:
            rename = {key: rename[key] for key  in rename.keys() if key in wt_df.columns}
            wt_df.rename(columns=self.rename, inplace=True)

        wt_df["Timestamp"] = pd.to_datetime(wt_df["Timestamp"], utc=True)
        f = ~wt_df["Timestamp"].isnull()
        wt_df = wt_df[f].sort_values(by = ["Timestamp"]).reset_index().drop(["index"], axis = 1) 

        # only keep the wanted columns (time, x and y features)
        wt_df = wt_df[[c for c in self.columns if c in wt_df.columns]]    
        return wt_df 
    
    def get_data_window(self, turbine, reset = False):
        t = turbine
        if not (os.path.isfile(os.path.join(self.PATH, f"{self.name}_{t}_X.pt")) and \
                                                                os.path.isfile(os.path.join(self.PATH, f"{self.name}_{t}_y.pt")) and \
                                                                os.path.isfile(os.path.join(self.PATH, f"{self.name}_{t}_timestamp.pkl")))\
                                                                 and not reset:
            self.X[t],self.y[t],self.timestamps[t] = self.make_window_data(self.in_shift, self.out_shift, self.in_cols, self.out_cols,
                                                                            turbine=[t], PATH = self.PATH, ret = True)
        else: 
            if self.X[t] is None or reset:
                self.X[t] = torch.load(os.path.join(self.PATH, f"{self.name}_{t}_X.pt"))
            if self.y[t] is None or reset:
                self.y[t]= torch.load(os.path.join(self.PATH, f"{self.name}_{t}_y.pt"))
            if self.timestamps[t] is None or reset:
                with open(os.path.join(self.PATH, f"{self.name}_{t}_timestamp.pkl"), 'rb') as f:
                    self.timestamps[t] = pickle.load(f)
           
        return self.X[t], self.y[t], self.timestamps[t]
    
    def feature_cut(self, feature_name, min = None, max = None, min_value = "drop", max_value = "drop", turbine_name = None, verbose = False):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name

        for t in turbine:
            assert feature_name in getattr(self, t).columns, print(f"Wrong feature name: {feature_name}")
            n = len(getattr(self, t)[feature_name])
            if min is not None:
                if min_value == "drop":
                    cut_offs = getattr(self, t)[getattr(self, t)[feature_name] < min].index
                    getattr(self, t).drop(cut_offs, inplace=True)
                else:
                    neg_powers = getattr(self, t)[getattr(self, t)[feature_name] < min].index
                    getattr(self, t).loc[neg_powers, feature_name] = min_value
            if max is not None: 
                if max_value == "drop":
                    cut_offs = getattr(self, t)[getattr(self, t)[feature_name] > max].index
                    getattr(self, t).drop(cut_offs, inplace=True)
                else:
                    neg_powers = getattr(self, t)[getattr(self, t)[feature_name] > max].index
                    getattr(self, t).loc[neg_powers, feature_name] = max_value 
            if verbose:
                print(f"Cutted {n-len(getattr(self, t)[feature_name])} element from {t} for {feature_name} in {self.name}")

    def line_cut(self, input_feature_name, output_feature_name, a, b, xmin, xmax, under = True, turbine_name = None):
        turbine = turbine_name if turbine_name is not None else self.list_turbine_name
        
        for t in turbine:
            assert input_feature_name in getattr(self, t).columns, print(f"Wrong feature name: {input_feature_name}")
            assert output_feature_name in getattr(self, t).columns, print(f"Wrong feature name: {output_feature_name}")
            if under:
                f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b < x[output_feature_name] or x[input_feature_name] >= xmax
            else:
                f = lambda x: x[input_feature_name] <= xmin or a*x[input_feature_name] + b > x[output_feature_name] or x[input_feature_name] >= xmax
            setattr(self, t, getattr(self, t)[getattr(self, t).apply(f, axis=1)])
            
    def feature_averaging(self, name, input_names):
        for t in self.list_turbine_name:
            assert all([name in getattr(self,t).columns for name in input_names]), print(f"The input names are incorrect: \
                                                                {[name for name in input_names if name not in getattr(self,t).columns]}")
            getattr(self, t)[name] = getattr(self, t)[input_names].mean(axis= 1)
                  
    def get_data(self, name, features, out_col = None):

        if isinstance(name, list):
            assert all([nm in self.list_turbine_name for nm in name]), print("Invalid name")
            data = pd.concat([getattr(self, t) for t in name])
            if out_col is None: 
                return data[features]
            else:
                return data[features], data[out_col]
        
        if name == "all":
            data = pd.concat([getattr(self, t) for t in self.list_turbine_name])
            return data[features] if out_col is None else data[features], data[out_col]
    
    def normalize_min_max(self, cols):
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c : (getattr(self, t)[c].max(), getattr(self, t)[c].min()) for c in cols}
            for c in cols:
                getattr(self, t)[c] = (getattr(self,t)[c] - self.normalized_dict[t][c][1])/(self.normalized_dict[t][c][0]-self.normalized_dict[t][c][1])

        with open(os.path.join(r'data_normalisation', f'min_max_normalisation_{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.normalized_dict, f)
    
    def normalize_mean_std(self, cols):
        for t in self.list_turbine_name:
            self.normalized_dict[t] = {c : (getattr(self, t)[c].mean(), getattr(self, t)[c].std()) for c in cols}
            for c in cols:
                getattr(self, t)[c] = (getattr(self,t)[c] - self.normalized_dict[t][c][0])/(self.normalized_dict[t][c][1])
 
        with open(os.path.join(r'data_normalisation', f'std_mean_normalisation_{self.name}.pkl'), 'wb') as f:
            pickle.dump(self.normalized_dict, f)

    def circ_embedding(self, cols):
        # Assuming that the unit is ° (0 to 360)
        for t in self.list_turbine_name:
            for c in cols:
                getattr(self, t)[f"{c}_cos"] = np.cos(getattr(self,t)[c]*2*np.pi/360)
                getattr(self, t)[f"{c}_sin"] = np.sin(getattr(self,t)[c]*2*np.pi/360)

    def drop_col(self, cols):
        for t in self.list_turbine_name:
            getattr(self,t).drop(columns = cols, inplace = True)

    def time_filling(self, method = 'linear', interpolation_limit = 6):
        for t in self.list_turbine_name:
            setattr(self,t, getattr(self, t).asfreq("10min"))
            getattr(self,t).interpolate(method = method, limit=interpolation_limit, inplace = True)

    def prepare_data(self): 
        # Data cleaning        
        if self.name == "Penmanshiel":
            self.feature_averaging("Gear_bear_temp_avg", ["Front_bear_temp", "Rotor_bear_temp", "Rear_bear_temp"])
            self.feature_averaging("Gen_bear_temp_avg", ["Gen_bear_front_temp", "Gen_bear_rear_temp"])
            self.feature_cut("Wind_speed", min=0, max= 25, min_value=0)
            self.feature_cut("Power", min=0, min_value=0)
            self.feature_cut("Rotor_speed", min=0, min_value=0)
            self.feature_cut("Gen_speed", min=0)
            self.feature_cut("Gen_speed", min = 0)
            self.feature_cut("Rotor_speed", max = 16)   
        
        if self.name == "Kelmarsh":
            self.feature_averaging("Gear_bear_temp_avg", ["Front_bear_temp", "Rotor_bear_temp", "Rear_bear_temp"])
            self.feature_averaging("Gen_bear_temp_avg", ["Gen_bear_front_temp", "Gen_bear_rear_temp"])
            self.feature_cut("Wind_speed", min=0, max= 25, min_value=0)
            self.feature_cut("Power", min=0,  min_value=0)
            self.feature_cut("Rotor_speed", min=0,  min_value=0)
            self.feature_cut("Gen_speed", min=0)
            self.line_cut("Wind_speed", "Power", a=0, b = 1900, xmin = 12, xmax = 25)
            self.line_cut("Wind_speed", "Power", a=170, b = -65, xmin = 10, xmax = 12)
        
        if self.name == "EDP":
            self.feature_averaging("Gen_bear_temp_avg", ["Gen_bearing_1_temp", "Gen_bearing_2_temp"])
            self.feature_cut("Wind_speed", min=0, max= 25, min_value=0)
            self.feature_cut("Power", min=0,  min_value=0)
            self.feature_cut("Rotor_speed", min=0,  min_value=0)
            self.feature_cut("Gen_speed", min=0)
            self.line_cut("Wind_speed", "Power", a = 0, b= 1100, xmin = 4.2, xmax= 25)
       
        
        cols = ["Power", "Wind_speed", "Wind_dir", "Amb_temp", "Gear_bear_temp_avg", "Rotor_speed", "Gen_speed", "Nacelle_position", "Gen_bear_temp_avg"]
        for  t in self.list_turbine_name:
            setattr(self, t, getattr(self, t)[~getattr(self, t).index.duplicated()])
        self.drop_col([c for c in getattr(self, self.list_turbine_name[0]).columns if c not in cols])
        self.time_filling(interpolation_limit = 12)

        # Normalization
        self.normalize_mean_std(cols=[c for c in cols if c in  ["Amb_temp", "Gear_bear_temp_avg", "Gen_bear_temp_avg"]])
        self.normalize_min_max(cols = [c for c in cols if c not in  ["Wind_dir", "Nacelle_position", "Amb_temp", "Gear_bear_temp_avg", "Gen_bear_temp_avg"]])
        self.circ_embedding(cols = [c for c in cols if c in  ["Wind_dir", "Nacelle_position"]])
        
        for t in self.list_turbine_name:
            getattr(self,t).dropna(how = "any", inplace = True)
            pd.to_datetime(getattr(self,t).index,errors='ignore') 
        self.save_data()

    def make_window_data(self, in_shift, in_cols, out_cols, turbine = None, PATH = r"data_prep", ret = False):
        turbine = turbine if turbine is not None else self.list_turbine_name 
        
        for t in turbine: 
            print(f"Preparing turbine {t}")
        
            data = getattr(self,t)
            in_data = data[in_cols].to_numpy(copy = True)
            out_data = data[out_cols].to_numpy(copy = True)
            
            index = data.index
            daterange = pd.date_range(start=index[0], end=index[-1], freq = "10min")

            #Creating blocks of timestamps without missing values
            blocks = [[]]
            i = 0
            for d in daterange:
                if d in data.index:
                    blocks[i].append(d)
                elif blocks[-1] != []:
                    blocks.append([])
                    i+=1
            if blocks[-1] == []:
                blocks = blocks[:-1]
            
            # For each block make a dataset
            in_datasets = []
            out_datasets = []
            timestamps = []

            for block in blocks:
                if len(block) > in_shift:
                    in_window = [in_data[i: i + in_shift +1] for i in range(len(block)-in_shift)]
                    out_window = [out_data[i+ in_shift] for i in range(len(block)-in_shift)]
                    timestamp = [block[i+in_shift] for i in range(len(block)-in_shift)]
                else:
                    in_window = None
                    out_window = None
                    timestamp = None
                if in_window is not None and out_window is not None and timestamp is not None:
                    in_datasets.append(in_window)
                    out_datasets.append(out_window)
                    timestamps.append(timestamp)

            # Concatenate the blocks
            if len(in_datasets) > 0:
                in_window_full = np.concatenate(in_datasets)
                out_window_full = np.concatenate(out_datasets)
                time = np.concatenate(timestamps)
                X, y = torch.Tensor(in_window_full), torch.Tensor(out_window_full)
                torch.save(X, os.path.join(PATH, f"{self.name}_{t}_X.pt"))
                torch.save(y, os.path.join(PATH, f"{self.name}_{t}_y.pt"))
                with open( os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl"), "wb") as f:
                    pickle.dump(time, f)
                
                if ret:
                    return X,y,time

    def get_window_data(self, in_shift, in_cols, out_cols, PATH = r"data_prep", RESET = False):
        
        if RESET:
             self.make_window_data(in_shift, in_cols, out_cols, turbine = None,  PATH = PATH)
        else:
            turbine = [t for t in self.list_turbine_name if not (os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_X.pt")) and \
                                                                 os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_y.pt")) and \
                                                                 os.path.isfile(os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl")))]
            
            self.make_window_data(in_shift, in_cols, out_cols, turbine= turbine, PATH= PATH)
            
        for t in self.list_turbine_name:
            setattr(self, f"{t}_X", torch.load(os.path.join(PATH, f"{self.name}_{t}_X.pt")))
            setattr(self, f"{t}_y", torch.load(os.path.join(PATH, f"{self.name}_{t}_y.pt")))
            with open(os.path.join(PATH, f"{self.name}_{t}_timestamp.pkl"), 'rb') as f:
                setattr(self, f"{t}_timestamp", pickle.load(f))

    def make_clients(self, turbine_names = "all", time_range = "all", data_threshold = 0, cols_num = "all", verbose = True):
        
        turbines = turbine_names if turbine_names != "all" else self.list_turbine_name
        X, y, timestamps = {}, {}, {}
        client_list = []

        for t in turbines:           
            X[t], y[t], timestamps[t] = self.get_data_window(t)
            
            if cols_num != "all":
                y[t] = y[t].squeeze(dim= 1)[:, cols_num]

            if time_range != "all":  
                # Filter the clients data to the provided time range       
                times = pd.to_datetime(timestamps[t])
                filt = (pd.to_datetime(time_range[0]) <= times) &  (pd.to_datetime(time_range[1]) >= times)
                X[t], y[t], timestamps[t] = X[t][filt] if X[t] is not None else [], y[t][filt] if y[t] is not None else [], \
                    timestamps[t][filt] if timestamps[t] is not None else []

            assert (len(X[t]) == len(y[t])), print(f"X and y must have same length, {len(X[t])} and {len(y[t])}")
            assert (len(X[t]) == len(timestamps[t])), print(f"X and timestamps must have same length, {len(X[t]) and {len(timestamps[t])}}")
            if len(X[t]) < data_threshold: # Minimum amount of data to consider t a valid client.
                continue  
            client = Client(name= f"{self.name}_{t}", farm_name = self.name, timestamps = timestamps[t], turbine_name = t)
            client._X, client._y = X[t], y[t]
            client.prepare_data(timestamps= timestamps[t])
            client_list.append(client)
            if verbose:
                print(f"Created client {self.name} turbine {t} with {len(X[t]) if X[t] is not None else 0}")

        return client_list

