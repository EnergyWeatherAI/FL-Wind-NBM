# Federated Learning for wind turbine condition monitoring. 

This project evaluates the effectiveness of Federated Learning in improving model performances compared to local turbine training while preserving the privacy of the data by not having to share the data of individual turbines. This code was developed by Albin Grataloup https://github.com/AlbinGr/

Official implementation for Wind turbine condition http://arxiv.org/abs/2409.03672
## Requirement 

Please install the packages in requirements.txt

## Load Data

Execute the following command to download and process the data:
``` python
py -m data.load_data
```

## Experiments

To recreate the experiment, run the Jupyter notebook experiment.ipynb. 
The experiment can be saved under a different experiment name by modifying the experiment_name variable. 

Then the results can be processed by running the notebook experiment_data.ipynb, using the corresponding experiment_name. 

The results can be obtained by running experiment_result_analysis.ipynb

## Results

The results from the paper can be directly visualized using experiment_result_analysis.ipynb with:
``` python
experiment_name = "Exp_1"
``` 
