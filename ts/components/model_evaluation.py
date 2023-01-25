import os
import sys
import torch
import numpy as np

from ts.entity.config_entity import ModelEvaluationConfig
from ts.entity.artifact_entity import ModelTrainerArtifacts, ModelEvaluationArtifacts
from ts.cloud_storage.s3_operations import S3Sync
from ts.logger import logging
from ts.exceptions import CustomException


class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts):
        self.model_evaluation_config = model_evaluation_config
        self.trainer_artifacts = model_trainer_artifacts

    def get_best_model_path(self):
        """
        It downloads the best model from the S3 bucket and returns the path of the best model
        
        Returns:
          The best model path is being returned.
        """
        try:
            model_path = self.model_evaluation_config.s3_model_path
            best_model_dir = self.model_evaluation_config.best_model_dir
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            s3_sync = S3Sync()
            best_model_path = None
            s3_sync.sync_folder_from_s3(
                folder=best_model_dir, aws_bucket_url=model_path)
            for file in os.listdir(best_model_dir):
                if file.endswith(".pt"):
                    best_model_path = os.path.join(best_model_dir, file)
                    logging.info(f"Best model found in {best_model_path}")
                    break
                else:
                    logging.info(
                        "Model is not available in best_model_directory")
            return best_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_model(self):
        """
        It loads the best model from S3, and then evaluates it on the validation set
        
        Returns:
          The best model path is being returned.
        """
        try:
            best_model_path = self.get_best_model_path()
            if best_model_path is not None:
                # load back the model
                state_dict = torch.load(best_model_path, map_location='cpu')
                loss = state_dict['loss']
                logging.info(f"S3 Model Validation loss is {loss}")
                logging.info(
                    f"Locally trained loss is {self.trainer_artifacts.model_loss}")
                s3_model_loss = loss
            else:
                logging.info(
                    "Model is not found on production server, So couldn't evaluate")
                s3_model_loss = None
            return s3_model_loss
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_evaluation(self):
        """
        It compares the loss of the model that was trained in the previous step 
        with the loss of the model that is already present in S3. 
        If the loss of the model that was trained in the previous step 
        is less than the loss of the model that is already present in S3, 
        then the model that was trained in the previous step is accepted as the best model
        """
        try:
            s3_model_loss = self.evaluate_model()
            tmp_best_model_loss = np.inf if s3_model_loss is None else s3_model_loss
            trained_model_loss = self.trainer_artifacts.model_loss
            evaluation_response = tmp_best_model_loss > trained_model_loss and trained_model_loss < self.model_evaluation_config.base_loss
            model_evaluation_artifacts = ModelEvaluationArtifacts(s3_model_loss=tmp_best_model_loss,
                                                                  is_model_accepted=evaluation_response,
                                                                  trained_model_path=os.path.dirname(
                                                                      self.trainer_artifacts.trained_model_path),
                                                                  s3_model_path=self.model_evaluation_config.s3_model_path
                                                                  )
            logging.info(
                f"Model evaluation completed! Artifacts: {model_evaluation_artifacts}")
            return model_evaluation_artifacts
        except Exception as e:
            raise CustomException(e, sys)