"""Every driver in `reproduction/experiments/` run end to end at a toy size, on a laptop, with no network.

These are not tests of the physics -- `src` is where that is checked -- they test that the wiring
holds: an argument that does not exist, a field renamed out from under a writer, a result whose keys
a reader cannot find. One module per subdirectory of `reproduction/experiments/`.

A package rather than a bare directory, so the `conftest.py` beside this file is imported as
`test_experiments.conftest`, distinct from the bare `conftest` of `tests/conftest.py`.
"""
