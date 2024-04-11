import pandas as pd
from model_parameter import *
from datetime import timedelta

BASE_PATH = r"results"

def add_weeks(start, week):
    end = pd.to_datetime(start) + timedelta(minutes=10*6*24*7*week)
    return end.strftime('%Y-%m-%d ') + '18:30:00'

class exp_param():
    def __init__(self, **args):
        key_list = ["max_week", "all_start_date", "all_cross_farm", "all_algorithm", "name", "cols", "test_type", "test_range", "data_threshold",\
                    "nb_week_test", "penmanshiel_turbine", "kelmarsh_turbine", "EDP_turbine", "min_week", "every_n_week"]
        
        self.list_farm_names = ["Penmanshiel", "Kelmarsh", "EDP"]
        self.max_week = 3
        self.min_week = 0
        self.every_n_week = 1
        self.all_start_date = ['2016-12-01 18:30:00', '2017-06-01 18:30:00']
        self.all_cross_farm = [False, True]
        self.all_algorithm = ["Local", "FedAvg", "FarmMixing"]
        self.name = "Exp_1"

        #Parameter for client selection
        self.cols = ["Gear_bear_temp_avg"]
        self.test_range= 6*24*7*4
        self.data_threshold = 144
        self.nb_week_test = 4
        self.penmanshiel_turbine = ["WT_11", "WT_12", "WT_13", "WT_14"]
        self.kelmarsh_turbine = ["WT_03", "WT_04", "WT_05", "WT_06"]
        self.EDP_turbine = "all"

        for key in key_list:
            if key in args.keys():
                setattr(self, key, args[key])
        
        self.nb_week = [i+1 for i in range(self.min_week, self.max_week, self.every_n_week)]
        self.all_dates_range = [(start, add_weeks(start, w)) for start in self.all_start_date for w in self.nb_week]

exp = exp_param()



