"""Dataset management endpoints."""


from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.api.dependencies import get_dataset_service
from app.core.exceptions import DatasetValidationError
from app.schemas.datasets import DatasetSummary
from app.services.dataset_service import DatasetService

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetSummary, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """Upload CSV dataset, validate schema, detect duplicates, leakage, and missing values."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are accepted.")

    content = await file.read()
    try:
        summary = dataset_service.process_and_save_csv(file.filename, content)
        return summary
    except DatasetValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset_metadata(
    dataset_id: str,
    dataset_service: DatasetService = Depends(get_dataset_service),
):
    """Retrieve full schema summary and validation diagnostics for a dataset."""
    try:
        return dataset_service.get_dataset_summary(dataset_id)
    except DatasetValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


@router.get("", response_model=list[dict])
def list_datasets(dataset_service: DatasetService = Depends(get_dataset_service)):
    """List all registered datasets in storage."""
    records = dataset_service.list_datasets()
    return [
        {
            "dataset_id": r.id,
            "filename": r.filename,
            "row_count": r.row_count,
            "column_count": r.column_count,
            "file_size_bytes": r.file_size_bytes,
            "suggested_target": r.suggested_target,
            "suggested_task": r.suggested_task,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in records
    ]
