"""Exceptions raised by the fusion package.

Both derive from ``ValueError`` so ``except ValueError`` keeps working.

* ``FusionInputError`` -- the *caller* passed something that cannot be fused
  (wrong shape, unaligned rows, unknown modality name, bad labels in ``fit``).
  This is an interface bug and should be fixed, not hidden.
* ``FusionConfigError`` -- an invalid or unsupported configuration.

Bad *model outputs* (NaN, negative, zero-sum probabilities, unusable quality
scores) are NOT exceptions by default: the affected modality is dropped and the
reason is reported in the result's ``status``/``warnings``. Set
``FusionConfig(strict=True)`` to turn those into ``FusionInputError`` as well.
"""


class FusionInputError(ValueError):
    """Invalid data passed to the fusion module."""


class FusionConfigError(ValueError):
    """Invalid or unsupported fusion configuration."""
