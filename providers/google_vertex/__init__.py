"""Google Vertex AI Express provider exports."""

from config.provider_catalog import GOOGLE_VERTEX_DEFAULT_BASE

from .client import VertexExpressProvider

__all__ = [
    "GOOGLE_VERTEX_DEFAULT_BASE",
    "VertexExpressProvider",
]