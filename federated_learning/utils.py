import sys
import os
from time import time
import functools
import pickle
from os.path import exists
import copy
from time import time
import torch

def memory_size(x):
    if torch.is_tensor(x):
        return x.element_size() * x.nelement() * 1e-06
    elif isinstance(x, dict):
        return sum([memory_size(x[key]) for key in x.keys()])
    else:
        return memory_size(torch.tensor(copy.deepcopy(x))) 

class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def execution_time(_func = None, verbose = True):
    def execution_time_decorator(func):
       
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time()
            a = func(*args, **kwargs)
            end = time()
            if verbose:
                print("Execution time of {} is {} second".format(func.__name__, round(end-start, 2)))
                return a
            else: 
                return end - start, a
        return wrapper
        
    if _func is None:
        return execution_time_decorator                     # 2
    else:
        return execution_time_decorator(_func) 
    

class timeout(object):
    def __init__(self, seconds):
        self.seconds = seconds
    def __enter__(self):
        self.die_after = time() + self.seconds
        return self
    def __exit__(self, type, value, traceback):
        pass
    @property
    def timed_out(self):
        return time() > self.die_after
    

class logs(dict):

    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __init__(self, **kwargs) -> None:
        self.__dict__ = {}
        pass
        
    def add_log(self, key, log, new = True):
        assert isinstance(key, str), print("key must be a str")
        if new: 
            assert key not in self.__dict__.keys(), print("Key already exists")
        self.__dict__[key] = log

    def add_value(self, key, value):
        assert isinstance(key, str), print("key must be a str")
        if key in self.__dict__.keys():
            self.__dict__[key].append(value)
        else:
            self.__dict__[key]= [value]

    def display(self, keys): 
        for key in keys:
            print(self.__dict__[key])

    def save(self, file_dir, file_name, verbose = False):
        if verbose:
            print("Saving {}".format(os.path.join(file_dir, file_name + ".pkl")))
        with open(os.path.join(file_dir, file_name + ".pkl"), 'wb') as fp:
            pickle.dump(self.__dict__, fp)
    
    def load(self, file_dir, file_name):
        print("Loading {}".format(os.path.join(file_dir, file_name + ".pkl")))
        #Load parameters 
        assert exists(os.path.join(file_dir, file_name + ".pkl")), print("No parameter file {}".format(file_dir + file_name))

        with open(os.path.join(file_dir, file_name + ".pkl"), 'rb') as fp:
            self.__dict__ = pickle.load(fp)
        return self
 

class Norm:

    def __init__(self, df=None):
        # Store the raw data.
        self.norm_dic = {}
        self.data = df
        
        if df is not None:
            self.add_data(df)

    def add_data(self, data):
        df = copy.deepcopy(data)
        for col in df.columns:
            self.norm_dic[col] = (df[col].mean(), df[col].std())
        

    def normalize(self, data):
        for col in data.columns:
            if col in self.norm_dic.keys():
                data[col] = (data[col] - self.norm_dic[col][0]) / self.norm_dic[col][1]
        return data

    def invert_normalize(self, data):
        dt = copy.deepcopy(data)
        for col in dt.columns:
            if col in list(self.norm_dic.keys()):
                dt[col] = dt[col] * self.norm_dic[1] + self.norm_dic[0]
            else:
                raise ValueError(
                    'The dataframe column ' + col + 'cannot be inverted because it is not in the normalization database')
            dt[col] = (dt[col] - dt[col].mean()) / dt[col].std()
        return dt

    def display(self):
        print(self.norm_dic)


class RBF(torch.nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None, device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels, device = device) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLoss(torch.nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY