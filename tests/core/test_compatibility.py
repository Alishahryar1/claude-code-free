import typing

import pytest

from core.compatibility import apply_compatibility_patches


def test_typing_eval_type_monkeypatch():
    """Verify that typing._eval_type monkeypatch handles prefer_fwd_module safely."""
    # This test should pass regardless of Python version because our patch
    # checks the signature before applying.

    _eval_type = getattr(typing, "_eval_type", None)
    if _eval_type is None:
        return  # Skip if not present in this Python version

    # Re-apply patches to ensure it's in a known state (it's idempotent)
    apply_compatibility_patches()

    # Get the (possibly patched) function
    patched_eval_type = getattr(typing, "_eval_type", None)
    if patched_eval_type is None:
        return

    # Verify we can call it with prefer_fwd_module even if original doesn't support it
    # We use a dummy ForwardRef or similar for testing if we wanted to be thorough,
    # but the goal here is to check the *argument handling*.

    # Check if the patch is actually applied by looking at signature of original
    # We can't easily get the *original* original here if it was already patched at import,
    # but we can verify the behavior of the current typing._eval_type.

    # Test call with prefer_fwd_module
    # We'll mock the internal call if we need to, but simply calling it with
    # the argument and seeing it doesn't raise TypeError is the goal.
    try:
        # We don't actually care about the result, just that it doesn't raise TypeError
        # for 'prefer_fwd_module'.
        # Since it might raise other errors for bad arguments, we catch everything.
        _eval_type_attr = "_eval_type"
        _eval_type_func = getattr(typing, _eval_type_attr)
        _eval_type_func(int, globals(), locals(), prefer_fwd_module=True)
    except TypeError as e:
        if "prefer_fwd_module" in str(e):
            pytest.fail(
                f"typing._eval_type still raises TypeError for prefer_fwd_module: {e}"
            )
    except Exception:
        # Other errors are fine, as long as it's not the TypeError we're fixing
        pass


def test_apply_compatibility_patches_idempotent():
    """Ensure apply_compatibility_patches can be called multiple times."""
    apply_compatibility_patches()
    apply_compatibility_patches()
    # No errors = success
