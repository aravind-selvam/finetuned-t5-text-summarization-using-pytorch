import os
import re
import sys

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from ts.cloud_storage.s3_operations import S3Sync
from ts.components.model_finetuner import T5smallFinetuner
from ts.constants import *
from ts.entity.config_entity import PredictionPipelineConfig
from ts.exceptions import CustomException
from ts.logger import logging


class SinglePrediction:
    def __init__(self):
        try:
            self.s3_sync = S3Sync()
            self.prediction_config = PredictionPipelineConfig()
            base_model_name = self.prediction_config.base_model_name
            self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        except Exception as e:
            raise CustomException(e, sys)

    def summarize_text(self, text, model):
        """
        > The function takes in a text and a model, and returns a summary of the text
        """
        def whitespace_handler(k): return re.sub(
            '\s+', ' ', re.sub('\n+', ' ', k.strip()))
        text_encoding = self.tokenizer(
            whitespace_handler(text),
            max_length=self.prediction_config.input_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        generated_ids = model.model.generate(
            input_ids=text_encoding['input_ids'],
            attention_mask=text_encoding['attention_mask'],
            max_length=self.prediction_config.output_max_length,
            num_beams=self.prediction_config.num_beams,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            early_stopping=True
        )
        preds = [self.tokenizer.decode(
            gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)for gen_id in generated_ids]
        return "".join(preds)

    def _get_model_in_production(self):
        """
        It checks if the model is available in the local directory, if not, it downloads the model from S3

        Returns:
          The model path is being returned.
        """
        try:
            s3_model_path = self.prediction_config.s3_model_path
            model_download_path = self.prediction_config.model_download_path
            os.makedirs(model_download_path, exist_ok=True)
            if len(os.listdir(model_download_path)) == 0:
                self.s3_sync.sync_folder_from_s3(
                    folder=model_download_path, aws_bucket_url=s3_model_path)
            for file in os.listdir(model_download_path):
                if file.endswith(".pt"):
                    prediction_model_path = os.path.join(
                        model_download_path, file)
                    logging.info(
                        f"Production model for prediction found in {prediction_model_path}")
                    break
                else:
                    logging.info(
                        "Model is not available in Prediction artifacts")
                    prediction_model_path = None
            return prediction_model_path
        except Exception as e:
            raise CustomException(e, sys)

    def prediction_step(self, input_text, model):
        """
        It takes in the input text and the model, and returns the summary of the input text

        Args:
          input_text: The text that you want to summarize.
          model: The model to be used for summarization.

        Returns:
          The summary of the input text.
        """
        try:
            with torch.no_grad():
                summary = self.summarize_text(text=input_text, model=model)
                logging.info("Summary of the input text: {}".format(summary))
                return summary
        except Exception as e:
            raise CustomException(e, sys)

    def get_model(self):
        """
        It loads the model from the path specified in the config file.

        Returns:
          The model is being returned.
        """
        try:
            prediction_model_path = self._get_model_in_production()
            if prediction_model_path is None:
                return None
            else:
                prediction_model = T5smallFinetuner(
                    model=self.base_model, tokenizer=self.tokenizer)
                model_state_dict = torch.load(
                    prediction_model_path, map_location='cpu')
                prediction_model.load_state_dict(
                    model_state_dict['model_state_dict'])
                prediction_model.eval()
            return prediction_model
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_text):
        """
        > The function takes in a string of text, and returns a list of predictions

        Args:
          input_text: The text that you want to predict the summary for.

        Returns:
          The generated predictions are being returned.
        """
        try:
            prediction_model = self.get_model()
            generated_predictions = self.prediction_step(input_text=input_text,
                                                         model=prediction_model)
            return generated_predictions
        except Exception as e:
            raise CustomException(e, sys)
