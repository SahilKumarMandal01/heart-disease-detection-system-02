from dataclasses import dataclass


@dataclass(frozen=True)
class DataIngestionArtifact:
    """
    Artifact produced by the Data Ingestion step.
    Contains file paths for raw, train, test, and schema outputs.
    """
    raw_file_path: str
    train_file_path: str
    test_file_path: str
    generated_schema: str

    def __str__(self) -> str:
        return (
            "\n📦 DataIngestionArtifact:\n"
            f"  ├── Raw File Path     : {self.raw_file_path}\n"
            f"  ├── Train File Path   : {self.train_file_path}\n"
            f"  ├── Test File Path    : {self.test_file_path}\n"
            f"  └── Schema File Path  : {self.generated_schema}\n"
        )


@dataclass(frozen=True)
class DataValidationArtifact:
    """
    Artifact produced by the Data Validation step.
    Contains the path to the validation report and the validation status flag.
    """
    report_file_path: str
    validation_status: bool

    def __str__(self) -> str:
        return (
            "\n📄 DataValidationArtifact:\n"
            f"  ├── Report File Path  : {self.report_file_path}\n"
            f"  └── Validation Status : {self.validation_status}\n"
        )


@dataclass(frozen=True)
class DataTransformationArtifact:
    """
    Artifact produced by the Data Transformation step.

    Stores absolute paths for:
    - The fitted preprocessing transformer
    - Transformed training dataset
    - Transformed testing dataset
    - Feature names file used during transformation
    """
    transformer_file_path: str
    transformed_train_file_path: str
    transformed_test_file_path: str
    feature_names_file_path: str

    def __str__(self) -> str:
        return (
            "\n🔧 DataTransformationArtifact:\n"
            f"  ├── Transformer File Path        : {self.transformer_file_path}\n"
            f"  ├── Transformed Train File Path  : {self.transformed_train_file_path}\n"
            f"  ├── Transformed Test File Path   : {self.transformed_test_file_path}\n"
            f"  └── Feature Names File Path      : {self.feature_names_file_path}\n"
        )


@dataclass(frozen=True)
class ModelTrainerArtifact:
    """
    Artifact produced by the Model Training step.

    Stores:
    - Path to the trained model
    - Path to model performance metrics
    """
    trained_model_file_path: str
    trained_model_metrics_file_path: str

    def __str__(self) -> str:
        return (
            "\n🤖 ModelTrainerArtifact:\n"
            f"  ├── Model File Path    : {self.trained_model_file_path}\n"
            f"  └── Metrics File Path  : {self.trained_model_metrics_file_path}\n"
        )
