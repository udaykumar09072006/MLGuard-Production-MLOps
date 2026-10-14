"""Domain-specific exceptions for MLGuard."""


class MLGuardException(Exception):
    """Base exception for all MLGuard errors."""
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DatasetValidationError(MLGuardException):
    """Raised when dataset fails schema, type, or integrity checks."""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class ModelTrainingError(MLGuardException):
    """Raised when training fails or receives invalid hyperparameters."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)


class ModelNotFoundError(MLGuardException):
    """Raised when requesting a model or version that does not exist."""
    def __init__(self, message: str):
        super().__init__(message, status_code=404)


class ModelNotActiveError(MLGuardException):
    """Raised when inference is requested but no active model is deployed."""
    def __init__(self, message: str = "No model version is currently activated for serving."):
        super().__init__(message, status_code=409)


class PredictionInputError(MLGuardException):
    """Raised when prediction input records are missing features or contain invalid types."""
    def __init__(self, message: str):
        super().__init__(message, status_code=422)


class MonitoringDriftError(MLGuardException):
    """Raised when drift calculation encounters invalid reference data or mismatch."""
    def __init__(self, message: str):
        super().__init__(message, status_code=400)
