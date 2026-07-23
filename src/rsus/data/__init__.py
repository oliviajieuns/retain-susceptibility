"""Dataset-neutral request objects and explicit adapter discovery."""

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.registry import (
    AdapterCapabilities,
    AdapterNotFoundError,
    AdapterRegistryError,
    DatasetAdapter,
    DatasetAdapterRegistry,
    get_adapter,
    register_adapter,
    registered_adapters,
)

__all__ = [
    "AdapterCapabilities",
    "AdapterNotFoundError",
    "AdapterRegistryError",
    "CandidateUniverse",
    "DatasetAdapter",
    "DatasetAdapterRegistry",
    "Example",
    "Request",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
