"""Dataset ingestion, file validation, and inspection service."""

import os

import pandas as pd
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.exceptions import DatasetValidationError
from app.db.repositories import DatasetRepository
from app.ml.data_validation import inspect_dataset
from app.schemas.datasets import DatasetSummary
from app.utils.file_storage import secure_save_path
from app.utils.identifiers import calculate_file_hash, generate_unique_id


class DatasetService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = DatasetRepository(db)

    def process_and_save_csv(self, filename: str, content: bytes) -> DatasetSummary:
        """Validate uploaded CSV content, save to controlled storage, and persist metadata."""
        if not content:
            raise DatasetValidationError("Uploaded file is empty (0 bytes).")

        size = len(content)
        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size > max_bytes:
            raise DatasetValidationError(f"File size ({size / (1024*1024):.1f}MB) exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit.")

        dataset_id = generate_unique_id("ds")
        target_name = f"{dataset_id}_{filename}"
        target_path = secure_save_path(settings.DATA_STORAGE_DIR, target_name)

        with open(target_path, "wb") as f:
            f.write(content)

        fingerprint = calculate_file_hash(target_path)

        try:
            df = pd.read_csv(target_path)
        except Exception as e:
            if os.path.exists(target_path):
                os.remove(target_path)
            raise DatasetValidationError(f"Failed to parse CSV file: {str(e)}")

        summary = inspect_dataset(
            df=df,
            dataset_id=dataset_id,
            filename=filename,
            file_path=target_path,
            file_size=size,
            fingerprint=fingerprint,
        )

        self.repo.save_dataset(summary)
        return summary

    def get_dataset_summary(self, dataset_id: str) -> DatasetSummary:
        record = self.repo.get_dataset(dataset_id)
        if not record:
            raise DatasetValidationError(f"Dataset '{dataset_id}' not found.")

        df = pd.read_csv(record.file_path)
        return inspect_dataset(
            df=df,
            dataset_id=record.id,
            filename=record.filename,
            file_path=record.file_path,
            file_size=record.file_size_bytes,
            fingerprint=record.fingerprint_sha256,
        )

    def load_dataset_dataframe(self, dataset_id: str) -> pd.DataFrame:
        record = self.repo.get_dataset(dataset_id)
        if not record:
            # Check if dataset_id matches sample data
            sample_housing = os.path.join(settings.SAMPLE_DATA_DIR, "housing_prices.csv")
            sample_churn = os.path.join(settings.SAMPLE_DATA_DIR, "customer_churn.csv")
            if dataset_id == "housing_prices" and os.path.exists(sample_housing):
                return pd.read_csv(sample_housing)
            if dataset_id == "customer_churn" and os.path.exists(sample_churn):
                return pd.read_csv(sample_churn)
            raise DatasetValidationError(f"Dataset '{dataset_id}' not found.")

        if not os.path.exists(record.file_path):
            raise DatasetValidationError(f"Dataset file at {record.file_path} is missing from disk.")
        return pd.read_csv(record.file_path)

    def list_datasets(self):
        return self.repo.list_datasets()
