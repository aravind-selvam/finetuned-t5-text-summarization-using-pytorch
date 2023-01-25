import os

ARTIFACTS_DIR: str = "artifacts"
SOURCE_DIR_NAME: str = 'ts'

BUCKET_NAME: str = "news_data"
FILE_NAME: str = "news_summary.csv"
S3_BUCKET_DATA_URI = "s3://news-data/raw_data/"

# common files
METADATA_DIR = "metadata"
METADATA_FILE_NAME: str = "metadata.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# constants related to data ingestion
DATA_INGESTION_ARTIFACTS_DIR: str = "data_ingestion_artifacts"
RAW_DATA_DIR_NAME: str = "raw_data"
DATA_INGESTION_TRAIN_DIR: str = "train"
DATA_INGESTION_TEST_DIR: str = "test"

# constants related to model training
MODEL_TRAINING_ARTIFACTS_DIR: str = "model_training_artifacts"
TRAINED_MODEL_NAME: str = 'model.pt'
CHECKPOINT_DIR: str = 'checkpoint'
LEARNING_RATE: float = 2e-5
EPOCHS: int = 1
BATCH_SIZE = 4
NUM_WORKERS = 0

# constants related to model evaluation
S3_BUCKET_MODEL_URI: str = "s3://news-data/model/"
MODEL_EVALUATION_DIR: str = "model_evaluation"
S3_MODEL_DIR_NAME: str = "s3_model"
IN_CHANNELS: int = 1
BASE_LOSS: float = 10

# constants related to model pusher
MODEL_PUSHER_DIR: str = "model_pusher"

# constants related to pipeline
COLUMNS_NAME =  ['text', 'ctext']
PRETRAINED_MODEL_NAME = "t5-small"

# constants related to prediction
PREDICTION_PIPELINE_DIR_NAME = "prediction_artifacts"
PREDICTION_MODEL_DIR_NAME = "prediction_model"