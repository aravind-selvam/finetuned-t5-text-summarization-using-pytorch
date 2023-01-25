import os
import sys
import pandas as pd
from dateutil import parser
from sklearn.model_selection import StratifiedShuffleSplit

from ts.logger import logging
from ts.exceptions import CustomException
from ts.entity.config_entity import DataIngestionConfig
from ts.entity.artifact_entity import DataIngestionArtifacts
from ts.constants import *
from ts.cloud_storage.s3_operations import S3Sync


class DataIngestion:
    """Ingest the data to the pipeline."""

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.s3_sync = S3Sync()
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_from_cloud(self) -> None:
        """
        > This function downloads the data from the s3 bucket to the local machine
        """
        try:
            logging.info("Initiating data download from s3 bucket...")
            self.raw_data_dir = self.data_ingestion_config.raw_data_dir
            self.s3_sync.sync_folder_from_s3(
                    folder=self.raw_data_dir, aws_bucket_url=S3_BUCKET_DATA_URI)
            logging.info(
                    f"Data is downloaded from s3 bucket to Download directory: {self.raw_data_dir}.")
        except Exception as e:
            raise CustomException(e, sys)

    def train_test_split(self) -> None:
        """
        It reads the raw data, converts the date column to month and year format, and then stratified
        shuffle split the data into train and test set
        """
        try:
            for file in os.listdir(self.raw_data_dir):
                if file.endswith('.csv'):
                    raw_data = pd.read_csv(os.path.join(self.raw_data_dir,file), encoding='latin-1')

            # instantiate shuffle split function
            split = StratifiedShuffleSplit(
                n_splits=5, train_size=0.85, test_size=0.15, random_state=42)
            
            # convert date(string) column to month and year format for stratified shuffle split
            for i in range(len(raw_data['date'])):
                raw_data['date'][i] = parser.parse(raw_data['date'][i]).strftime("%m-%Y")
            
            # stratified split using 
            for train_index, test_index in split.split(raw_data, raw_data['date']):
                strat_train_set = raw_data.loc[train_index]
                strat_test_set = raw_data.loc[test_index]
            
            # save train and test to specified path
            train_file_path = self.data_ingestion_config.train_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            # create train and test directory if it doesn't exist
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)

            # save train and test set
            strat_train_set.to_csv(train_file_path, index=False)
            strat_test_set.to_csv(test_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        > This function is used to download the data from the cloud and split it into train and test data
        """
        logging.info("Starting data ingestion component...")
        try:
            self.get_data_from_cloud()
            self.train_test_split()
            data_ingestion_artifacts = DataIngestionArtifacts(raw_file_dir=self.data_ingestion_config.raw_data_dir,
                                                             train_file_path= self.data_ingestion_config.train_file_path,
                                                             test_file_path= self.data_ingestion_config.test_file_path
                                                             )

            logging.info(f"Data ingestion artifact is generated {data_ingestion_artifacts}")
            
            logging.info("Data ingestion completed successfully... \
                        Note: If data is not downloaded try deleting the data folder and try again.")
            
            return data_ingestion_artifacts
        
        except Exception as e:
            logging.error(
                "Error in Data Ingestion component! Check above logs")
            raise CustomException(e, sys)