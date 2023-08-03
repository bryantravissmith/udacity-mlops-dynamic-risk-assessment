import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])


# Function for training the model
def train_model():

    model = LogisticRegression(
        C=1.0, class_weight=None, dual=False,
        fit_intercept=True, intercept_scaling=1, l1_ratio=None,
        max_iter=100,  n_jobs=None,
        penalty='l2', random_state=0, solver='liblinear',
        tol=0.0001, verbose=0, warm_start=False)

    data = pd.read_csv(f"{dataset_csv_path}/finaldata.csv")
    y = data.pop('exited')
    x = data[['lastmonth_activity', 'lastyear_activity',
              'number_of_employees']]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.20,
                                              random_state=42)

    model.fit(x_train, y_train)

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    filehandler = open(f'{model_path}/trainedmodel.pkl', 'wb')
    pickle.dump(model, filehandler)


if __name__ == '__main__':
    train_model()
