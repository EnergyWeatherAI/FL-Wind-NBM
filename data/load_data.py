import sys , os, copy, glob

sys.path.insert(0, os.path.join(os.getcwd(), "federated_learning"))
sys.path.insert(0, os.path.join(os.getcwd(), "data"))
os.chdir(os.path.join(os.getcwd(), "data"))
import pandas as pd
from data.Data import data_farm

import requests
from xlsx2csv import Xlsx2csv

import wget
import zipfile
import pathlib


import shutil

if __name__ == "main":

    paths = [".temp", "data_normalisation", "data_prep", "EDP", "Penmanshiel", "Kelmarsh"]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)


    # Penmanshiel and Kelmarsh
            
    # Download the data
    for farm in ["Penmanshiel", "Kelmarsh"]:
        list_of_pages = open(f'WGET_{farm}.txt', 'r+')
        for link in list_of_pages.readlines():
            name = link.split("/")[-1]
            print("\n" + name.strip())
            file = wget.download("\n" + link.strip(), out=r".temp")
        list_of_pages.close()    
        print("\n")

    for farm in ["Penmanshiel", 'Kelmarsh']:
        for f in glob.glob(f'.temp/{farm}_SCADA_*.zip'):
            with zipfile.ZipFile(f, 'r') as archive:
                    # Extract all contents of the Zip file to a directory with the same name as the file
                    archive.extractall(path = os.path.join(pathlib.Path(f".temp/{farm}"), f'{pathlib.Path(f).stem}'))
                    # Print a message indicating that the extraction is complete
                    print(f"Extracted contents from '{pathlib.Path(f).name}' to '{pathlib.Path(f).stem}' directory.")
                
    def merge_csvs(csv_folder_path, farm, wt_id = "1"):
        print(f"Finding, Processing, and Merging all .csvs for WT_ID: {wt_id}")
        wt_dfs = []
        p = csv_folder_path.glob(f"*/Turbine_Data_{farm}_{wt_id}*.csv")
        wt_csvs = [x for x in p if x.is_file()]
        for i, f in enumerate(wt_csvs):
            print(f"Loading file {i+1}/{len(wt_csvs)}")
            df = pd.read_csv(f, skiprows=9, parse_dates=["# Date and time"])
            lost_production_columns = [col for col in df.columns if "Lost Production" in col]
            no_loss_df = df
            for col in lost_production_columns:
                losses = no_loss_df[no_loss_df[col] > 0].index
                no_loss_df.drop(losses, inplace=True)


            # drop non-average columns (the rest may not be needed now, but maybe later)
            non_avgs = ['Gear oil inlet pressure, Max (bar)', 'Gear oil inlet pressure, Min (bar)',
                'Gear oil inlet pressure, StdDev (bar)', 'Gear oil pump pressure, Max (bar)',
                'Gear oil pump pressure, Min (bar)', 'Gear oil pump pressure, StdDev (bar)',
                'Blade angle (pitch position) A, Max (°)',
                'Blade angle (pitch position) A, Min (°)',
                'Blade angle (pitch position) A, Standard deviation (°)',
                'Blade angle (pitch position) B, Max (°)',
                'Blade angle (pitch position) B, Min (°)',
                'Blade angle (pitch position) B, Standard deviation (°)',
                'Blade angle (pitch position) C, Max (°)',
                'Blade angle (pitch position) C, Min (°)',
                'Blade angle (pitch position) C, Standard deviation (°)',
                'Yaw bearing angle, Max (°)',
                'Yaw bearing angle, Min (°)',
                'Yaw bearing angle, StdDev (°)',
                'Motor current axis 1, Max (A)',
                'Motor current axis 1, Min (A)',
                'Motor current axis 1, StdDev (A)',
                'Motor current axis 2, Max (A)',
                'Motor current axis 2, Min (A)',
                'Motor current axis 2, StdDev (A)',
                'Motor current axis 3, Max (A)',
                'Motor current axis 3, Min (A)',
                'Motor current axis 3, StdDev (A)',
                'Current L1 / U, min (A)',
                'Current L1 / U, max (A)',
                'Current L1 / U, StdDev (A)',
                'Current L2 / V, max (A)',
                'Current L3 / W, max (A)',
                'Current L2 / V, min (A)',
                'Current L2 / V, StdDev (A)',
                'Current L3 / W, min (A)',
                'Current L3 / W, StdDev (A)',
                'Grid voltage, Max (V)',
                'Grid voltage, Min (V)',
                'Grid voltage, Standard deviation (V)',
                'Voltage L1 / U, Min (V)',
                'Voltage L1 / U, Max (V)',
                'Voltage L1 / U, Standard deviation (V)',
                'Voltage L2 / V, Min (V)',
                'Voltage L2 / V, Max (V)',
                'Voltage L2 / V, Standard deviation (V)',
                'Voltage L3 / W, Min (V)',
                'Voltage L3 / W, Max (V)',
                'Voltage L3 / W, Standard deviation (V)',
                'Temperature motor axis 1, Max (°C)',
                'Temperature motor axis 1, Min (°C)',
                'Temperature motor axis 1, StdDev (°C)',
                'Temperature motor axis 2, Max (°C)',
                'Temperature motor axis 2, Min (°C)',
                'Temperature motor axis 2, StdDev (°C)',
                'Temperature motor axis 3, Max (°C)',
                'Temperature motor axis 3, Min (°C)',
                'Temperature motor axis 3, StdDev (°C)',
                'Vane position 1+2, Max (°)',
                'Vane position 1+2, Min (°)',
                'Vane position 1+2, StdDev (°)',
                'Wind speed Sensor 2, Standard deviation (m/s)',
                'Wind speed Sensor 2, Minimum (m/s)',
                'Wind speed Sensor 2, Maximum (m/s)',
                'Wind speed Sensor 1, Standard deviation (m/s)',
                'Wind speed Sensor 1, Minimum (m/s)',
                'Wind speed Sensor 1, Maximum (m/s)',
                'Apparent power (kVA)',
                'Apparent power, Max (kVA)',
                'Apparent power, Min (kVA)',
                'Apparent power, StdDev (kVA)',
                'Cable windings from calibration point',
                'Metal particle count',
                'Metal particle count counter',
                'Cable windings from calibration point, Max',
                'Cable windings from calibration point, Min',
                'Cable windings from calibration point, StdDev',
                'Drive train acceleration (mm/ss)',
                'Tower Acceleration X (mm/ss)',
                'Tower Acceleration y (mm/ss)',
                'Tower Acceleration X, Min (mm/ss)',
                'Tower Acceleration X, Max (mm/ss)',
                'Tower Acceleration Y, Min (mm/ss)',
                'Tower Acceleration Y, Max (mm/ss)',
                'Drive train acceleration, Max (mm/ss)',
                'Drive train acceleration, Min (mm/ss)',
                'Drive train acceleration, StdDev (mm/ss)',
                'Tower Acceleration X, StdDev (mm/ss)',
                'Tower Acceleration Y, StdDev (mm/ss)',
                'Grid frequency, Max (Hz)',
                'Grid frequency, Min (Hz)',
                'Grid frequency, Standard deviation (Hz)',
                'Time-based Contractual Avail.',
                'Time-based IEC B.2.2 (Users View)',
                'Time-based IEC B.2.3 (Users View)',
                'Time-based IEC B.2.4 (Users View)',
                'Time-based IEC B.3.2 (Manufacturers View)',
                'Production-based IEC B.2.2 (Users View)',
                'Production-based IEC B.2.3 (Users View)',
                'Production-based IEC B.3.2 (Manufacturers View)',
                'Time-based System Avail.',
                'Production-based System Avail.',
                'Production-based Contractual Avail.',
                'Time-based System Avail. (Planned)',
                'Production-based System Avail. (virtual)',
                'Time-based Contractual Avail. (Global)',
                'Time-based Contractual Avail. (Custom)',
                'Production-based Contractual Avail. (Global)',
                'Production-based Contractual Avail. (Custom)',
                ]

            no_loss_df.drop(columns=lost_production_columns)
            no_loss_df.drop(columns=non_avgs, inplace=True)
            wt_dfs.append(no_loss_df)

    
                
        # merge
        scada_df = pd.concat(wt_dfs) 
        scada_df.sort_values(by='# Date and time', ascending = True, inplace = True) 
        return scada_df

    all_turbine_Penmanshiel = ["01", "02", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15"]
    all_turbine_Kelmarsh = ["1", "2", "3", "4", "5", "6"]
    farms = ["Penmanshiel", "Kelmarsh"]
    print("\n Penmanshiel")
    for t in all_turbine_Penmanshiel:
        df = merge_csvs(pathlib.Path(f".temp/Penmanshiel"), "Penmanshiel", wt_id = t)
        df.to_csv(f"Penmanshiel/WT_{t}.csv", index=False)
        print(f"Penmanshiel Wind turbine WT {t} stored\n")
    print("\n Kelmarsh")
    for t in all_turbine_Kelmarsh:
        df = merge_csvs(pathlib.Path(f".temp/Kelmarsh"), "Kelmarsh", wt_id = t)
        df.to_csv(f"Kelmarsh/WT_0{t}.csv", index=False)
        print(f"Kelmarsh Wind turbine WT {t} stored")

    #EDP 
        print("\n Downloading EDP data")
    url_EDP_2016 = r"https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2016.xlsx"
    url_EDP_2017 = r"https://www.edp.com/sites/default/files/2023-04/Wind-Turbine-SCADA-signals-2017_0.xlsx"

    r_EDP_2016 = requests.get(url_EDP_2016, allow_redirects=True)
    open(r'.temp\EDP_2016.xlsx', 'wb').write(r_EDP_2016.content)
    r_EDP_2017 = requests.get(url_EDP_2017, allow_redirects=True)
    open(r'.temp\EDP_2017.xlsx', 'wb').write(r_EDP_2017.content)


    print("Convert 2016")
    Xlsx2csv(r'.temp\EDP_2016.xlsx', outputencoding="utf-8").convert(r'.temp\EDP_2016.csv')
    print("Convert 2017")
    Xlsx2csv(r'.temp\EDP_2017.xlsx', outputencoding="utf-8").convert(r'.temp\EDP_2017.csv')

    df_2016 = pd.read_csv(".temp/EDP_2016.csv", parse_dates= ["Timestamp"])
    df_2017 = pd.read_csv(".temp/EDP_2017.csv", parse_dates= ["Timestamp"])

    df = pd.concat([df_2016, df_2017])
    df["Timestamp"].unique(), df_2016["Timestamp"].unique(),df_2017["Timestamp"].unique()

    df = pd.concat([df_2016, df_2017])
    for t in df["Turbine_ID"].unique():
        d = copy.deepcopy(df[df["Turbine_ID"] == t])
        tnum = t[-2:]
        print(f"Saving EDP wind turbine WT_{tnum}")
        d.to_csv(f"EDP/WT_{tnum}.csv", index=False)

    shutil.rmtree('.temp/')


    print("\n\n Processing the data\n")
    # Penmanshiel
    cols_penmanshiel = ["Timestamp", "Power", "Power_std", "Wind_speed", "Wind_speed_std", "Wind_dir",  "Wind_dir_std", "Amb_temp", "Amb_temp_std","Gear_oil_temp" , "Gear_oil_temp_std",
            "Front_bear_temp", "Front_bear_temp_std", "Rotor_speed", "Rotor_speed_std", "Gen_speed", "Gen_speed_std", "Transf_temp", "Transf_temp_std", "Blade_pitch", 
            "Blade_pitch_std", "Nacelle_position", "Nacelle_position_std", "Rotor_bear_temp", "Rotor_bear_temp_std", "Rear_bear_temp", "Rear_bear_temp_std",
            "Gen_bear_rear_temp", "Gen_bear_rear_temp_std", "Gen_bear_front_temp", "Gen_bear_front_temp_std", "Blade angle (pitch position) A (°)", \
            "Blade angle (pitch position) B (°)", "Blade angle (pitch position) C (°)"]
    rename_penmanshiel = {"# Date and time": "Timestamp", 
                "Power (kW)": "Power", 
                "Power, Standard deviation (kW)": "Power_std",
                "Wind speed (m/s)": "Wind_speed",
                "Wind speed, Standard deviation (m/s)": "Wind_speed_std",
                "Wind direction (°)": "Wind_dir",
                "Wind direction, Standard deviation (°)" : "Wind_dir_std",
                "Nacelle ambient temperature (°C)": "Amb_temp",
                "Nacelle ambient temperature, StdDev (°C)": "Amb_temp_std",
                "Gear oil temperature (°C)" : "Gear_oil_temp",
                "Gear oil temperature, Standard deviation (°C)" : "Gear_oil_temp_std",
                "Front bearing temperature (°C)": "Front_bear_temp", 
                "Front bearing temperature, Standard deviation (°C)": "Front_bear_temp_std",
                "Rotor speed (RPM)": "Rotor_speed", 
                "Rotor speed, Standard deviation (RPM)": "Rotor_speed_std",
                "Generator RPM (RPM)": "Gen_speed", 
                "Generator RPM, Standard deviation (RPM)": "Gen_speed_std",
                "Generator bearing rear temperature (°C)": "Gen_bear_rear_temp", 
                "Generator bearing rear temperature, Std (°C)": "Gen_bear_rear_temp_std", 
                "Generator bearing front temperature (°C)": "Gen_bear_front_temp",
                "Generator bearing front temperature, Std (°C)" : "Gen_bear_front_temp_std",
                "Transformer temperature (°C)": "Transf_temp",
                "Transformer temperature, StdDev (°C)" : "Transf_temp_std",
                "Blade angle (pitch position) (°)" : "Blade_pitch",
                "Blade angle (pitch position), Standard deviation (°)" : "Blade_pitch_std",
                "Nacelle position (°)" : 'Nacelle_position',
                "Nacelle position, Standard deviation (°)" : "Nacelle_position_std" ,
                "Rotor bearing temp (°C)": "Rotor_bear_temp",
                "Rotor bearing temp, StdDev (°C)" : "Rotor_bear_temp_std",
                "Rear bearing temperature (°C)" : "Rear_bear_temp",
                "Rear bearing temperature, Standard deviation (°C)" : "Rear_bear_temp_std"
                }
    list_names_penmanshiel = ["WT_01", "WT_02", "WT_15", "WT_04", "WT_05", "WT_06", "WT_07", "WT_08", "WT_09", "WT_10", "WT_11", "WT_12", "WT_13", "WT_14"]
    DATA_PATH_PENMANSHIEL = r'Penmanshiel'

    # Kelmarsh 
    cols_kelmarsh  = ["Timestamp", "Power", "Power_std", "Wind_speed", "Wind_speed_std", "Wind_dir",  "Wind_dir_std", "Amb_temp", "Amb_temp_std","Gear_oil_temp" , "Gear_oil_temp_std",
            "Front_bear_temp", "Front_bear_temp_std", "Rotor_speed", "Rotor_speed_std", "Gen_speed", "Gen_speed_std", "Transf_temp", "Transf_temp_std", "Blade_pitch", 
            "Blade_pitch_std", "Nacelle_position", "Nacelle_position_std", "Rotor_bear_temp", "Rotor_bear_temp_std", "Rear_bear_temp", "Rear_bear_temp_std",
            "Gen_bear_rear_temp", "Gen_bear_rear_temp_std", "Gen_bear_front_temp", "Gen_bear_front_temp_std", "Blade angle (pitch position) A (°)", \
            "Blade angle (pitch position) B (°)", "Blade angle (pitch position) C (°)"]
    rename_kelmarsh = {"# Date and time": "Timestamp", 
                "Power (kW)": "Power", 
                "Power, Standard deviation (kW)": "Power_std",
                "Wind speed (m/s)": "Wind_speed",
                "Wind speed, Standard deviation (m/s)": "Wind_speed_std",
                "Wind direction (°)": "Wind_dir",
                "Wind direction, Standard deviation (°)" : "Wind_dir_std",
                "Nacelle ambient temperature (°C)": "Amb_temp",
                "Nacelle ambient temperature, StdDev (°C)": "Amb_temp_std",
                "Gear oil temperature (°C)" : "Gear_oil_temp",
                "Gear oil temperature, Standard deviation (°C)" : "Gear_oil_temp_std",
                "Front bearing temperature (°C)": "Front_bear_temp", 
                "Front bearing temperature, Standard deviation (°C)": "Front_bear_temp_std",
                "Rotor speed (RPM)": "Rotor_speed", 
                "Rotor speed, Standard deviation (RPM)": "Rotor_speed_std",
                "Generator RPM (RPM)": "Gen_speed", 
                "Generator RPM, Standard deviation (RPM)": "Gen_speed_std",
                "Generator bearing rear temperature (°C)": "Gen_bear_rear_temp", 
                "Generator bearing rear temperature, Std (°C)": "Gen_bear_rear_temp_std", 
                "Generator bearing front temperature (°C)": "Gen_bear_front_temp",
                "Generator bearing front temperature, Std (°C)" : "Gen_bear_front_temp_std",
                "Transformer temperature (°C)": "Transf_temp",
                "Transformer temperature, StdDev (°C)" : "Transf_temp_std",
                "Blade angle (pitch position) (°)" : "Blade_pitch",
                "Blade angle (pitch position), Standard deviation (°)" : "Blade_pitch_std",
                "Nacelle position (°)" : 'Nacelle_position',
                "Nacelle position, Standard deviation (°)" : "Nacelle_position_std" ,
                "Rotor bearing temp (°C)": "Rotor_bear_temp",
                "Rotor bearing temp, StdDev (°C)" : "Rotor_bear_temp_std",
                "Rear bearing temperature (°C)" : "Rear_bear_temp",
                "Rear bearing temperature, Standard deviation (°C)" : "Rear_bear_temp_std"
                }
    list_names_kelmarsh = ["WT_01", "WT_02", "WT_03", "WT_04", "WT_05", "WT_06"]
    DATA_PATH_KELMARSH = r'Kelmarsh'

    # EDP
    cols_edp = ["Timestamp", "Power", "Wind_speed",  "Wind_speed_std", "Wind_dir", "Amb_temp", "Gear_oil_temp",  "Gear_bear_temp_avg",  "Rotor_speed", \
            "Rotor_speed_std", "Gen_speed", "Gen_speed_std",  "Gen_bearing_1_temp",  "Gen_bearing_2_temp",  "Transf_temp_p1", "Transf_temp_p2", \
            "Transf_temp_p3", "Blade_pitch", "Blade_pitch_std", 'Nacelle_position']
    rename_edp = {"Timestamp": "Timestamp", 
                "Prod_LatestAvg_TotActPwr": "Power", 
                "Amb_WindSpeed_Avg": "Wind_speed",
                "Amb_WindSpeed_Std": "Wind_speed_std",
                "Amb_WindDir_Abs_Avg": "Wind_dir",
                "Amb_Temp_Avg": "Amb_temp",
                "Gear_Oil_Temp_Avg" : "Gear_oil_temp",
                "Gear_Bear_Temp_Avg": "Gear_bear_temp_avg", 
                "Rtr_RPM_Avg": "Rotor_speed", 
                "Rtr_RPM_Std": "Rotor_speed_std",
                "Gen_RPM_Avg": "Gen_speed", 
                "Gen_RPM_Std": "Gen_speed_std",
                "Gen_Bear_Temp_Avg": "Gen_bearing_1_temp", 
                "Gen_Bear2_Temp_Avg": "Gen_bearing_2_temp", 
                "HVTrafo_Phase1_Temp_Avg": "Transf_temp_p1",
                "HVTrafo_Phase2_Temp_Avg": "Transf_temp_p2",
                "HVTrafo_Phase3_Temp_Avg": "Transf_temp_p3",
                "Blds_PitchAngle_Avg" : "Blade_pitch",
                "Blds_PitchAngle_Std" : "Blade_pitch_std",
                "Nac_Direction_Avg" : 'Nacelle_position'
                }
    list_names_edp = ["WT_01", "WT_06", "WT_07", "WT_11"]
    DATA_PATH_EDP = r"EDP"

    # Initialize and prepare the data
    print("Penmanshiel")
    farm_penmanshiel = data_farm(cols = cols_penmanshiel, name = "Penmanshiel", list_turbine_name= list_names_penmanshiel, DATA_PATH=DATA_PATH_PENMANSHIEL,\
                                rename = rename_penmanshiel, time_name = "# Date and time", initialize= True)

    print("\nKelmarsh")
    farm_kelmarsh = data_farm(cols = cols_kelmarsh, name = "Kelmarsh", list_turbine_name= list_names_kelmarsh, DATA_PATH=DATA_PATH_KELMARSH,\
                                rename = rename_kelmarsh, time_name = "# Date and time", initialize= True)

    print("\nEDP")
    farm_edp = data_farm(cols = cols_edp, name = "EDP", list_turbine_name= list_names_edp, DATA_PATH=DATA_PATH_EDP,\
                                rename = rename_edp, time_name = "Timestamp", initialize= True)


    # Features of interest
    in_cols = ["Wind_speed", "Wind_dir_cos", "Wind_dir_sin", "Amb_temp"]
    out_cols = ["Power", "Gear_bear_temp_avg", "Rotor_speed", "Gen_speed","Gen_bear_temp_avg", "Nacelle_position_cos", "Nacelle_position_sin"]

    print("\n\n Preparing data windows\n")
    # Creating and storing time window datasets 
    print("Kelmarsh")
    farm_kelmarsh.get_window_data(in_shift=24*6, in_cols=in_cols, out_cols=out_cols, PATH = r"data_prep", RESET = True)

    print("\nPenmanshiel")
    farm_penmanshiel.get_window_data(in_shift=24*6, in_cols=in_cols, out_cols=out_cols, PATH = r"data_prep", RESET = True)

    print("\nEDP")
    farm_edp.get_window_data(in_shift=24*6, in_cols=in_cols, out_cols=out_cols, PATH = r"data_prep", RESET = True)