"""User-facing exception hierarchy for QuantForge."""


class QuantForgeError(Exception):
    """Base exception for all QuantForge errors."""


class ConfigError(QuantForgeError):
    """Configuration validation or loading error."""


class DatasetError(QuantForgeError):
    """Dataset loading or processing error."""


class DatasetColumnError(DatasetError):
    """Dataset column not found error."""


class ModelError(QuantForgeError):
    """Model creation or loading error."""


class QuantizationError(QuantForgeError):
    """Quantization strategy error."""


class CheckpointError(QuantForgeError):
    """Checkpoint save/load error."""


class TrackingError(QuantForgeError):
    """Experiment tracking error."""
