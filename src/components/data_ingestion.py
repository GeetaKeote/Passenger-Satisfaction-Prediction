import sys,os
from src.exception import CustomException
from src.logger import logging

from src.constant import *
from src.config.configuration import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTrainer

class DataIngestionconfig():
    raw_data_path:str = RAW_FILE_PATH
    train_data_path:str = TRAIN_FILE_PATH
    test_data_path:str = TEST_FILE_PATH


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config=DataIngestionconfig()


    def initiate_Data_ingestion(self):
        logging.info("data ingestion started")
        logging.info(f"data set path : {DATASET_PATH}")

        try:

            df=pd.read_csv(DATASET_PATH)
            

            logging.info("dataset columns :{df.columns} ")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path,index=False)

            logging.info("dataset in raw data dir" )



            logging.info("train test split")

            train_set,test_set = train_test_split(df,test_size=0.20,random_state=42)

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok =True)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            logging.info(f"train data path : {TRAIN_FILE_PATH}")


            os.makedirs(os.path.dirname(self.data_ingestion_config.test_data_path),exist_ok =True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info(f"train data path : {TEST_FILE_PATH}")


            logging.info(f"data ingestion done " )

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("exception occur at data ingestion stage")
            raise CustomException(e,sys)
        

if __name__ == "__main__":
    obj=DataIngestion()
    train_data_path,test_data_path = obj.initiate_Data_ingestion()

    data_transformation = DataTransformation()    
    train_arr, test_arr, _ = data_transformation.inititate_data_transformation(train_data_path, test_data_path)

    modeltrainer = ModelTrainer()
    print(modeltrainer.inititate_model_trainer(train_arr, test_arr))

# src/components/data_ingestion.py