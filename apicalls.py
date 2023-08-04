import json
import os
import requests

# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
URL = 'http://127.0.0.1:8000'

# Call each API endpoint and store the responses
response1 = requests.post(f'{URL}/prediction', data={
    'data_path': 'testdata/testdata.csv'
}).text
response2 = requests.get(f'{URL}/scoring').text
response3 = requests.get(f'{URL}/summarystats').text
response4 = requests.get(f'{URL}/diagnostics').text

# combine all API responses
responses = '\n'.join([ response1, response2, response3, response4])

# write the responses to your workspace
filehandler = open(f'{dataset_csv_path}/apireturns_2.txt', 'w')
filehandler.write(responses)

