import os  # for interacting with the operating system (i.e. train and test path)
import sys # for implementing logger and exception file
from src.logger import logging
from src.exception import CustomException
import pandas as pd  # for reading the dataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass  # a placeholder which creates a class itself


## intialize the data ingestion configuration
@dataclass
class DataIngestionconfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'raw.csv')


# data ingestion class
class DataIngestion():
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data', 'gemstone.csv'))
            logging.info('Reading the Dataset as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info('Raw data created')

            logging.info('Train-test split')
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        

        except Exception as e:
            logging.info('Error occured in Data Ingestion Config')
            raise CustomException(e, sys)








    
    