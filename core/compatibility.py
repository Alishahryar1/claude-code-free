import inspect
import logging
import typing

logger = logging.getLogger(__name__)


def apply_compatibility_patches():
    """Apply compatibility patches for different Python/Library versions."""

    # Patch for Python 3.14 compatibility with Pydantic 2.13.3
    # Pydantic 2.13.3 assumes typing._eval_type has a 'prefer_fwd_module' argument
    # which was likely in a dev version of 3.14 but missing in 3.14.0rc1+.
    try:
        _original_eval_type = getattr(typing, "_eval_type", None)
        if _original_eval_type is None:
            return

        # Check if it already lacks prefer_fwd_module
        sig = inspect.signature(_original_eval_type)
        if "prefer_fwd_module" not in sig.parameters:

            def _patched_eval_type(*args, **kwargs):
                kwargs.pop("prefer_fwd_module", None)
                return _original_eval_type(*args, **kwargs)

            target_attr = "_eval_type"
            setattr(typing, target_attr, _patched_eval_type)
            logger.debug(
                "Applied Python 3.14 typing._eval_type monkeypatch for Pydantic"
            )
    except AttributeError, ValueError:
        # If typing._eval_type doesn't exist or signature can't be read, skip
        pass


apply_compatibility_patches()
