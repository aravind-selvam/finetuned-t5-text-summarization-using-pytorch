from dataclasses import dataclass

# Data ingestion artifacts
@dataclass
class DataIngestionArtifacts:
    raw_file_dir: str
    train_file_path: str
    test_file_path: str

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    model_loss: float

@dataclass
class ModelEvaluationArtifacts:
    s3_model_loss: float
    is_model_accepted: bool
    trained_model_path: str
    s3_model_path: str

@dataclass
class ModelPusherArtifacts:
    response: dict