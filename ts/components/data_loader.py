from ts.components.news_dataset import NewsDataset
import pytorch_lightning as pl
import pandas as pd
from torch.utils.data import DataLoader
from ts.constants import *


class NewsDataLoader(pl.LightningDataModule):
    def __init__(self, train_file_path, val_file_path, tokenizer, batch_size,
                 columns_name, source_len=1024, target_len=128, corpus_size=500):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.batch_size = batch_size
        self.nrows = corpus_size
        self.columns_name = columns_name
        self.target_len = target_len
        self.source_len = source_len

    def prepare_data(self):
        """
        It reads the data from the csv file, and then drops the rows with missing values
        """
        train_data = pd.read_csv(self.train_file_path,
                                 nrows=self.nrows/0.80, encoding='latin-1')
        val_data = pd.read_csv(
            self.val_file_path, nrows=self.nrows/0.20, encoding='latin-1')
        train_data = train_data[self.columns_name]
        val_data = val_data[self.columns_name]
        self.train_data = train_data.dropna()
        self.val_data = val_data.dropna()

    def setup(self, stage=None):
        """
        It takes the training and validation data, and converts it into a format that can be used by the
        model
        
        Args:
          stage: The stage of the pipeline. This is used to determine which data to use.
        """
        X_train = self.train_data.iloc[:, -2].values
        y_train = self.train_data.iloc[:, -1].values
        X_val = self.val_data.iloc[:, -2].values
        y_val = self.val_data.iloc[:, -1].values

        self.train_dataset = (X_train, y_train)
        self.val_dataset = (X_val, y_val)

    def train_dataloader(self):
        train_data = NewsDataset(source_texts=self.train_dataset[0],
                                 target_texts=self.train_dataset[1],
                                 tokenizer=self.tokenizer,
                                 source_len=self.source_len,
                                 target_len=self.target_len
                                 )
        return DataLoader(train_data, self.batch_size, num_workers=NUM_WORKERS, shuffle=True, pin_memory=True)

    def val_dataloader(self):
        val_data = NewsDataset(source_texts=self.val_dataset[0],
                               target_texts=self.val_dataset[1],
                               tokenizer=self.tokenizer,
                               source_len=self.source_len,
                               target_len=self.target_len
                               )
        return DataLoader(val_data, self.batch_size, num_workers=NUM_WORKERS, pin_memory=True)