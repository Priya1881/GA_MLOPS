import yaml
import os
#import json
import pickle
import pandas as pd
from application_logging.logger import App_Logger
from prediction_service.preprocess_prediction import preprocessor

params_path = "params.yaml"
schema_path = os.path.join("prediction_service", "schema_in.json")
file_object = open("Prediction_log.txt",'+a')
logger=App_Logger()

class NotAValidFilename(Exception):
    def __init__(self, message="Not a Valid File Name"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path=params_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(path):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    preprocessor(path)
    data=pd.read_csv(config['split_data']['test_path'])
    model = pickle.load(open(model_dir_path,'rb'))
    prediction = model.predict(data)

    for i in range(0,len(prediction)):
        if prediction[i] < 1:
            prediction[i] = 0

    pred_values = pd.DataFrame(prediction,columns=['transactionRevenue(in USD)'])
    pred_values=pred_values.applymap(lambda x:'$'+str(round(x,2)))
    logger.log(file_object,'$ is added in front of transaction values')

    pred_values.to_csv('prediction_batch_files/outputfiles/predicted_file.csv',index=False)
    response = 'The predictions are in generated at prediction_bath_files/outputfiles/predicted_file.csv'
    return (response)


def validate_input(path):
    if path != "test_data.csv":
        raise NotAValidFilename
    else:
        return True

def form_response(path):
    if validate_input(path):
        response = predict(path)
        return response


def api_response(path):
    try:
        if validate_input(path):
            preprocessor(path)
            response = "Result : The Predicted values are in the file "
            return response

    except Exception as e:
        response = {"response": str(e)}
        return response
