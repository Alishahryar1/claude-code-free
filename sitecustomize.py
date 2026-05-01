"""Interpreter startup tweaks for the repository.

Python automatically imports ``sitecustomize`` (if present on ``sys.path``)
during interpreter startup.

We use it to apply a small Python 3.14 compatibility shim required for the
current FastAPI/Pydantic dependency set.
"""

from core.typing_compat import patch_typing_eval_type_for_pydantic

patch_typing_eval_type_for_pydantic()
