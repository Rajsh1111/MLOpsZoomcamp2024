
import os
import sys
import pickle
import pandas as pd
from datetime import datetime
from sklearn.feature_extraction import DictVectorizer
from datetime import datetime




def read_dataframe(filename:str):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    categorical = ['PULocationID', 'DOLocationID']

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    
    return dicts



def load_model():
    with open('models/lin_reg.bin', 'rb') as f_in:
     dv, model = pickle.load(f_in)
     return dv, model

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['tpep_pickup_datetime'] = df['tpep_pickup_datetime']
    df_result['PULocationID'] = df['PULocationID']
    df_result['DOLocationID'] = df['DOLocationID']
    df_result['actual_duration'] = df['duration']
    df_result['predicted_duration'] = y_pred
    #df_result['predicted_duration1'] = y_pred1
    df_result['diff'] = df_result['actual_duration'] - df_result['predicted_duration']

    df_result.to_parquet(output_file, index=False)


def apply_model(input_file, output_file):


    print(f'reading the data from {input_file}...')
    df = read_dataframe(input_file)
    dicts = prepare_dictionaries(df)

    print(f'loading the model...')
    dv,model = load_model()
    x_val = dv.transform(dicts)
    print(f'applying the model...')
    y_pred = model.predict(x_val)
    #y_pred1 = model.predict(dicts)

    print(f'saving the result to {output_file}...')

    save_results(df, y_pred,output_file)
    return output_file



def run():
    taxi_type = sys.argv[1] # 'yellow'
    year = int(sys.argv[2]) # 2023
    month = int(sys.argv[3]) # 4
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'


    apply_model(
        input_file=input_file,
        output_file=output_file)
  


if __name__ == '__main__':
    run()








