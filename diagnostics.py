import pandas as pd
import timeit
import os
import json
import pickle
import subprocess

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
test_data_path = os.path.join(config['test_data_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# Function to get model predictions
def model_predictions(datapath):
    filehandler = open(f'{prod_deployment_path}/trainedmodel.pkl', 'rb')
    model = pickle.load(filehandler)
    data = pd.read_csv(datapath)
    y_pred = model.predict(data[['lastmonth_activity', 'lastyear_activity',
                                 'number_of_employees']])
    return list(y_pred)


# Function to get summary statistics
def dataframe_summary():
    dataframe = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')[[
        'lastmonth_activity', 'lastyear_activity', 'number_of_employees'
    ]]
    means = list(dataframe.mean())
    medians = list(dataframe.median())
    stdevs = list(dataframe.std())
    return [means, medians, stdevs]


# Function to get NAs rates
def dataframe_nas():
    dataframe = pd.read_csv(f'{dataset_csv_path}/finaldata.csv')[[
        'lastmonth_activity', 'lastyear_activity', 'number_of_employees'
    ]]
    nas = list(dataframe.isna().mean())
    return nas


# Function to get timings
def execution_time():
    start_time = timeit.default_timer()
    os.popen('python ingestion.py')
    ingesting_time = timeit.default_timer() - start_time

    start_time = timeit.default_timer()
    os.popen('python training.py')
    training_time = timeit.default_timer() - start_time
    return [ingesting_time, training_time]


# Function to check dependencies
def outdated_packages_list():
    outdated = subprocess.check_output([
        'python', '-m', 'pip', 'list', '--outdated', '--format', 'columns'
    ]).decode('utf-8').split('\n')

    outdated_df = pd.DataFrame(columns=['package', 'installed', 'current'])
    for package in outdated[2:]:
        clean_info = [s for s in package.split(' ') if s != ''][:3]
        if len(clean_info) == 3:
            outdated_df = outdated_df.append(
                pd.DataFrame([clean_info],
                             columns=['package', 'installed', 'current'])
            )
    return outdated_df


def run():
    model_predictions(f'{test_data_path}/testdata.csv')
    dataframe_summary()
    dataframe_nas()
    execution_time()
    outdated_packages_list()


if __name__ == '__main__':
    run()