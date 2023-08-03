import pickle
import os
import json

# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# function for deployment
def store_model_into_pickle(model):
    if not os.path.exists(prod_deployment_path):
        os.mkdir(prod_deployment_path)

    os.popen(f'cp {dataset_csv_path}/latestscore.txt' + ' ' +
             f'{prod_deployment_path}/latestscore.txt')
    os.popen(f'cp {dataset_csv_path}/ingestedfiles.txt' + ' ' +
             f'{prod_deployment_path}/ingestedfiles.txt')
    filehandler = open(f'{prod_deployment_path}/trainedmodel.pkl', 'wb')
    pickle.dump(model, filehandler)


if __name__ == '__main__':
    filehandler = open('practicemodels/trainedmodel.pkl', 'rb')
    model = pickle.load(filehandler)
    store_model_into_pickle(model)
