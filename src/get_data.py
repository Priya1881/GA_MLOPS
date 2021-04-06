import yaml
import pandas as pd
import argparse

def read_params(config_path):
    with open(config_path) as yaml_file:
        config=yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config=read_params(config_path)
    train_data_path=config['data_source']['train_csv']
    test_data_path=config['data_source']['test_csv']
    df = pd.read_csv(train_data_path)
    df1 = pd.read_csv(test_data_path)
    return (df,df1)


if __name__ == "__main__":
     args=argparse.ArgumentParser()
     args.add_argument("--config",default="params.yaml")
     parsed_args = args.parse_args()
     train_df,test_df=get_data(config_path=parsed_args.config)
     print(test_df.head())