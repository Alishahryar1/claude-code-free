"""Request metrics tracking service.

Tracks the number of API requests per model per hour.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from loguru import logger


class MetricsService:
    """Tracks and aggregates request metrics by model and hour."""

    def __init__(self):
        # Structure: {model_id: {hour_key: request_count}}
        self._metrics: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._lock = None  # For thread-safety if needed

    @staticmethod
    def _get_hour_key(dt: datetime | None = None) -> str:
        """Get hour key in format 'YYYY-MM-DD-HH' (UTC)."""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d-%H")

    def record_request(self, model_id: str) -> None:
        """Record a single request for a model.

        Args:
            model_id: The Claude model ID (e.g., "claude-opus-4-20250514")
        """
        if not model_id:
            model_id = "unknown"

        hour_key = self._get_hour_key()
        self._metrics[model_id][hour_key] += 1
        logger.debug(f"Metrics recorded: model={model_id}, hour={hour_key}")

    def get_metrics(
        self, hours: int = 24, model_id: str | None = None
    ) -> dict[str, Any]:
        """Get aggregated metrics.

        Args:
            hours: Number of recent hours to include (default 24)
            model_id: Filter to specific model, or None for all models

        Returns:
            Dictionary with metrics grouped by model and hour
        """
        result = {}

        # Determine which models to include
        models_to_include = (
            [model_id] if model_id and model_id in self._metrics else self._metrics.keys()
        )

        for model in sorted(models_to_include):
            model_data = {}

            # Get all hours for this model
            for hour_key in sorted(self._metrics[model].keys()):
                model_data[hour_key] = self._metrics[model][hour_key]

            if model_data:
                result[model] = model_data

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": result,
            "summary": self._compute_summary(result),
        }

    def get_hourly_summary(self, hours: int = 24) -> dict[str, Any]:
        """Get hourly summary across all models.

        Args:
            hours: Number of recent hours to include

        Returns:
            Dictionary with total requests per hour
        """
        hourly_totals: dict[str, int] = defaultdict(int)

        for model_id, hour_data in self._metrics.items():
            for hour_key, count in hour_data.items():
                hourly_totals[hour_key] += count

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hourly": {
                hour_key: count for hour_key, count in sorted(hourly_totals.items())
            },
        }

    def get_model_summary(self) -> dict[str, Any]:
        """Get summary by model (total requests per model)."""
        model_totals: dict[str, int] = {}

        for model_id, hour_data in self._metrics.items():
            model_totals[model_id] = sum(hour_data.values())

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": {
                model_id: count
                for model_id, count in sorted(
                    model_totals.items(), key=lambda x: x[1], reverse=True
                )
            },
            "total_requests": sum(model_totals.values()),
        }

    def _compute_summary(self, metrics: dict[str, dict[str, int]]) -> dict[str, Any]:
        """Compute summary statistics from metrics."""
        total_requests = 0
        model_count = len(metrics)
        hour_count = 0
        requests_per_model = {}

        for model_id, hour_data in metrics.items():
            model_total = sum(hour_data.values())
            requests_per_model[model_id] = model_total
            total_requests += model_total

            if hour_count == 0:
                hour_count = len(hour_data)

        return {
            "total_requests": total_requests,
            "model_count": model_count,
            "hour_count": hour_count,
            "requests_per_model": requests_per_model,
        }

    def reset_metrics(self) -> None:
        """Clear all metrics (for testing or maintenance)."""
        self._metrics.clear()
        logger.info("Metrics cleared")


# Global metrics instance
_metrics_service = MetricsService()


def get_metrics_service() -> MetricsService:
    """Get the global metrics service instance."""
    return _metrics_service
