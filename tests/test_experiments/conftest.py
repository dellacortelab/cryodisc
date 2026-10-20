"""The pair every driver test is run on. The loader itself is `conftest.load_driver`."""

import pytest
from conftest import hinged_pair_files


@pytest.fixture(scope="module")
def separable_pair(tmp_path_factory):
    """A pair carrying enough signal for the benchmark to have something to measure, as two PATHS.

    Unlike `toy_pair`, this pair has a real, signed axis, so a purity line fitted to it is not
    fitting noise. The same geometry the preprocessing suites triage and template against, from
    `conftest.hinged_coordinates`. Paths and not a `ConformerPair`, because a driver takes files.
    """
    return hinged_pair_files(tmp_path_factory.mktemp("separable"))
