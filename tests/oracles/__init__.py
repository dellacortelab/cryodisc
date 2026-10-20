"""Independent reference implementations, and the tests that compare production against them.

Nothing in a `ref_*` module imports `cryodisc`. Every routine is written from the
mathematics by a deliberately different method than production (derived wavelength vs. fit, adaptive
quadrature vs. fixed-grid trapezoid, explicit mode sum vs. FFT), so a disagreement is between two
derivations rather than a restatement of one implementation.

Tolerances are set by the looser of the two sides -- production's float32 against the float64 oracle
-- and noted wherever the dtype is what sets them.
"""
