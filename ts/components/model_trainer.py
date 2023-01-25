import os, sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from ts.entity.config_entity import ModelTrainerConfig
from ts.entity.artifact_entity import ModelTrainerArtifacts
from ts.exceptions import CustomException
from ts.logger import logging
from ts.exceptions import CustomException


class ModelTrainer:
    """
    Model Trainer
    """
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                model, dataloader):
        self.model_trainer_config = model_trainer_config
        self.model = model
        self.dataloader = dataloader

    def get_trainer_setup(self):
        """
        This function creates a checkpoint callback, a tensorboard logger and an early stopping callback
        """
        try:
            logging.info("get_trainer_setup method started")
            checkpoint_dir = self.model_trainer_config.checkpoint_dir
            checkpoint_fname = self.model_trainer_config.checkpoint_fname
            self.checkpoint_callback = ModelCheckpoint(
                dirpath= checkpoint_dir,
                filename= checkpoint_fname,
                save_top_k=1,
                verbose=True,
                monitor='val_loss',
                mode= 'min'
            )
            logging.info("custom callbacks created")

            self.lighting_logger = TensorBoardLogger("lighting_logs", name='summarizer_model')
            self.early_stopping_callback = EarlyStopping(monitor='val_loss', patience=1, verbose=False, mode='min')
            logging.info('early_stopping_callback created')

            logging.info("get_trainer_setup method completed")
        except Exception as e:
            raise CustomException(e, sys)

    def get_trainer_object(self):
        """
        It creates a trainer object from the Pytorch Lightning library. 
        
        The trainer object is used to train the model.
        
        Returns:
          The trainer object is being returned.
        """
        try:
            logging.info("get_trainer_object method started")
            max_epochs = self.model_trainer_config.epochs

            if torch.cuda.is_available():
                trainer = pl.Trainer(check_val_every_n_epoch=1, max_epochs=max_epochs, accelerator='gpu',logger= self.lighting_logger,
                                    callbacks=[self.early_stopping_callback, self.checkpoint_callback]
                                    )
            else:
                trainer = pl.Trainer(check_val_every_n_epoch=1, max_epochs=max_epochs,logger= self.lighting_logger,
                                    callbacks=[self.early_stopping_callback, self.checkpoint_callback]
                                    )
            logging.info("get_trainer_object method completed")

            return trainer
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_model_trainer(self):
        """
        It takes the model and dataloader from the model trainer config and trains the model using the
        trainer object
        
        Returns:
          ModelTrainerArtifacts object
        """
        try:
            logging.info("Model training started")
            self.get_trainer_setup()
            trainer = self.get_trainer_object()

            torch.cuda.empty_cache()
            trainer.fit(self.model, self.dataloader)
            logging.info("model fine tuning done")
            
            trained_model_loss = trainer.callback_metrics
            trained_model_loss = float(trained_model_loss['val_loss'])
            trained_model_path = self.model_trainer_config.trained_model_dir
            os.makedirs(os.path.dirname(trained_model_path), exist_ok=True)
            torch.save({'model_state_dict': self.model.state_dict(),
                        'loss': trained_model_loss}, trained_model_path)
            model_trainer_artifacts = ModelTrainerArtifacts(model_loss=trained_model_loss, 
                                                            trained_model_path=trained_model_path
                                                            )
            logging.info(f"model trainder artifact created {model_trainer_artifacts}")
            logging.info("Model training completed")
            return model_trainer_artifacts
        except Exception as e:
            raise CustomException(e, sys)