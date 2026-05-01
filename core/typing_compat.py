"""Small runtime compatibility shims.

Python 3.14 changed the private signature of ``typing._eval_type`` and removed
the ``prefer_fwd_module`` keyword argument.

Pydantic (and therefore FastAPI) may still pass that keyword, which causes an
import-time crash. We apply a small wrapper to drop the unsupported kwarg.
"""

import inspect
import typing
from collections.abc import Callable
from functools import wraps
from typing import Any, cast

_TYPING_EVAL_TYPE_PATCHED = False


def patch_typing_eval_type_for_pydantic() -> None:
    """Allow Pydantic to run on Python 3.14 by dropping ``prefer_fwd_module``.

    Python 3.14 removed ``prefer_fwd_module`` from ``typing._eval_type``.
    Pydantic 2.13.3 still passes it, raising ``TypeError`` during FastAPI
    import. This function wraps ``typing._eval_type`` to accept and ignore the
    keyword.
    """

    global _TYPING_EVAL_TYPE_PATCHED
    if _TYPING_EVAL_TYPE_PATCHED:
        return

    eval_type_obj = getattr(typing, "_eval_type", None)
    if eval_type_obj is None:
        _TYPING_EVAL_TYPE_PATCHED = True
        return

    try:
        sig = inspect.signature(eval_type_obj)
    except TypeError, ValueError:
        return

    if "prefer_fwd_module" in sig.parameters:
        _TYPING_EVAL_TYPE_PATCHED = True
        return

    original_eval_type = cast(Callable[..., Any], eval_type_obj)

    @wraps(original_eval_type)
    def _eval_type_patched(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("prefer_fwd_module", None)
        return original_eval_type(*args, **kwargs)

    typing.__dict__["_eval_type"] = _eval_type_patched
    _TYPING_EVAL_TYPE_PATCHED = True
