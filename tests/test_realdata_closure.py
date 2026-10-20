"""The real-data arm graded on a NUMBER, not on bookkeeping.

`test_realdata_drivers.py` checks only shapes, so a wrong-frequency CTF or a constant fraction offset
passes it. This plants a known mixture -- two conformers at a known occupancy, written to `.mrcs`
under a RELION 3.1 project and read back through `score_pool` -- and asserts a pool planted at
fraction `f` reads back at `f`, within 3 sigma of the estimator's own error bar
(`statistics.power.fraction_error`). `d^2` is checked against a floor too, so a bug that destroys the
separation cannot widen the error bar until any answer fits.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import hinged_pair_files, load_driver, write_relion_star

from cryodisc import CollectionPlan
from cryodisc.grids.poses import pose_bank
from cryodisc.particles.sim import simulate_stack
from cryodisc.resolution import noise_spectrum
from cryodisc.specimen.structures import align_conformers
from cryodisc.statistics.composition import fraction_no_pool
from cryodisc.statistics.information import deflection
from cryodisc.statistics.power import fraction_error

BOX_SIZE = 48
PIXEL_SIZE = 1.5
DOSE = 100.0
RMSF = 0.5
NODES = 54
DEFOCUS_RANGE = (8000.0, 20000.0)
"""A real spread, so `transfer_from_meta` has to group rather than build one CTF for everything."""

HINGE = 16.0
"""How far the blob's positive-x half swings, in A. Turned up until the conformational change is
resolvable at a particle count a test can afford; the geometry is otherwise the driver default."""

N_PARTICLES = 1000
"""Particles per pool. `sigma` scales as `1/sqrt(N)`; this is where 3 sigma passes under the
constant fraction offset this test exists to catch."""

SEED = 5
D2_FLOOR = 1.0
"""A wide margin below the pool's separation as a deflection; its job is to fail loudly
when that separation collapses rather than let the derived error bar swallow the failure."""


def plant_pool(directory: Path, pair_paths: tuple[Path, Path], fraction: float):
    """Simulate a pool of known composition and write it as a deposition would ship it.

    Poses are drawn ON the quadrature nodes the driver marginalizes over, which is what isolates the
    COMPOSITION estimator from the pose bank's coverage -- the same choice
    `reproduction/experiments/benchmark/known_fraction.py --poses bank` makes, and for the same reason.

    The particles are split across TWO `.mrcs` files with alternating rows, so the driver has to
    restore the metadata's row order: a reader that returned images grouped by file would pair every
    particle with another one's defocus and state, which no shape check can see.

    Args:
        directory: Where the star file and the project root are written.
        pair_paths: The two structure files.
        fraction: Share of the pool put in state A.

    Returns:
        `(star_path, project_root, stack)`, with `stack` carrying the truth -- `stack.fraction(0)` is
        the occupancy actually DRAWN, which is the planted number and not the one asked for.
    """
    import mrcfile

    pair = align_conformers(pair_paths[0], pair_paths[1], assembly=None)
    plan = CollectionPlan(dose=DOSE, pixel_size=PIXEL_SIZE, defocus_range=DEFOCUS_RANGE)
    grid = pose_bank(NODES)
    pose_key, stack_key = jax.random.split(jax.random.key(SEED))
    stack = simulate_stack(
        pair.models(plan, box_size=BOX_SIZE),
        key=stack_key,
        n_particles=N_PARTICLES,
        noise_spectrum=noise_spectrum(plan, BOX_SIZE, rmsf=RMSF),
        occupancy=(fraction, 1.0 - fraction),
        plan=plan,
        angles=grid.angles[jax.random.randint(pose_key, (N_PARTICLES,), 0, grid.angles.shape[0])],
    )

    root = directory / "project"
    extract = root / "Extract" / "job005"
    extract.mkdir(parents=True)
    images = np.asarray(jnp.fft.ifft2(stack.images).real, np.float32)
    names = [""] * N_PARTICLES
    for half in (0, 1):
        rows = np.arange(half, N_PARTICLES, 2)
        with mrcfile.new(extract / f"stack_{half}.mrcs", overwrite=True) as handle:
            handle.set_data(images[rows])
        for frame, row in enumerate(rows):
            names[row] = f"{frame + 1:06d}@Extract/job005/stack_{half}.mrcs"

    # the optics recorded are the ones the particles were rendered through, because the whole
    # measurement is the driver rebuilding this CTF from the file and matching it against the images
    star = write_relion_star(
        directory / "particles.star",
        names,
        np.asarray(stack.defocus, float),
        pixel_size=PIXEL_SIZE,
        box_size=BOX_SIZE,
        optics={
            "voltage_in_kv": plan.voltage_in_kv,
            "spherical_aberration_in_mm": plan.spherical_aberration_in_mm,
            "amplitude_contrast_ratio": plan.amplitude_contrast_ratio,
        },
    )
    return star, root, stack


@pytest.fixture(scope="module")
def separable_pair(tmp_path_factory):
    """The two conformers as PATHS, bent far enough to be resolvable at 1000 particles. See `HINGE`."""
    return hinged_pair_files(tmp_path_factory.mktemp("closure_pair"), hinge=HINGE)


@pytest.mark.parametrize("fraction", [0.30, 0.70])
def test_a_pool_planted_at_a_known_fraction_reads_back_at_it_through_the_real_driver(
    separable_pair, tmp_path, fraction
):
    """The real-data arm's validation loop, closed: a mixture of known composition written to disk in
    the formats a deposition ships, read back through `score_pool`, and graded on the fraction rather
    than on the shape of what came out.

    Two planted fractions on opposite sides of the midpoint, because a single point cannot separate an
    offset from a compression of the axis and 0.5 would be invariant under a swapped pair of banks.
    Catches what the shape-only test does not: a CTF at the wrong spatial frequencies, which
    decorrelates every template and takes `d^2` to zero, and any constant offset in the recovered
    fraction.
    """
    pytest.importorskip("mrcfile")
    driver = load_driver("realdata/score_pool.py")
    star, root, stack = plant_pool(tmp_path, separable_pair, fraction)
    planted = float(stack.fraction(0))

    out, log_ratio = tmp_path / "pool.json", tmp_path / "lambda.npy"
    argv = [
        "--meta", str(star), "--root", str(root),
        "--structure-a", str(separable_pair[0]), "--structure-b", str(separable_pair[1]), "--assembly", "none",
        "--out", str(out), "--log-ratio", str(log_ratio),
        "--box-size", str(BOX_SIZE), "--cap-bank", "504",  # marginal arm converges; one cheap bank
    ]  # fmt: skip
    assert driver.main(argv) == 0

    result = json.loads(out.read_text())["result"]
    assert result["n_particles"] == N_PARTICLES
    assert result["n_defocus_groups"] > 1
    # the marginal arm converged the pose bank rather than fixing a node count
    assert result["convergence"]["status"] in ("converged", "censored")
    assert result["convergence"]["nodes_realized"] == max(int(k) for k in result["convergence"]["curve"])

    measured = jnp.asarray(np.load(log_ratio))
    state = np.asarray(stack.state)
    # the planted labels set the error bar: d^2 becomes sigma, so it is asserted against a floor of
    # its own, or an estimator that separated nothing would be graded against an unbounded error bar
    d_squared = float(deflection(measured[state == 0], measured[state == 1]))
    assert d_squared > D2_FLOOR, f"the pool did not separate: d^2 = {d_squared:.3g}"

    recovered = fraction_no_pool(measured)
    sigma = float(fraction_error(d_squared, n_particles=N_PARTICLES, fraction=planted))
    assert abs(recovered - planted) <= 3.0 * sigma, (
        f"planted {planted:.4f}, recovered {recovered:.4f}, "
        f"off by {(recovered - planted) / sigma:+.2f} sigma at sigma = {sigma:.4f}"
    )
