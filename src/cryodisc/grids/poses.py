"""Quadrature grids for the pose latents.

A pose grid is a quadrature RULE: summing without the weights tilts every integral toward wherever
the grid is dense, which on a viewing sphere is the poles. The volume is rotated by R and projected
along lab z, so the direction looked down is R^T z; under intrinsic zyz that depends on (phi, theta)
alone and psi only spins the image. The Haar measure sin(theta) dtheta dphi dpsi puts Gauss-Legendre
nodes in cos(theta), with uniform grids on the periodic phi and psi. Log weights are normalized, so a
log-likelihood stays comparable across grid refinements.
"""

from functools import cache

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from cryojax.rotations import SO3
from jax.scipy.special import logsumexp
from jaxtyping import Array, Complex, Float
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from cryodisc.grids.fourier import phase_ramp


class PoseGrid(eqx.Module):
    """Quadrature nodes on SO(3).

    Args:
        angles: (n, 3) Euler angles in degrees, cryoJAX intrinsic zyz, ready for `forward_batch`.
        log_weights: Log quadrature weight per node, normalized to sum to one.
    """

    angles: Float[Array, "n 3"]
    log_weights: Float[Array, " n"]


class ShiftGrid(eqx.Module):
    """Candidate translations and their log prior, on a lattice.

    A quadrature rule like `PoseGrid`. `rungs.marginal` additionally needs whole-pixel shifts (it
    scores every candidate from one inverse transform) and checks them rather than trusting a field.
    For a single continuous translation, call `fourier.phase_ramp` directly.

    Args:
        shifts: (n, 2) translations as (y, x) in pixels.
        log_weights: Log prior per candidate, normalized to sum to one.
        box_size: Image size the phase ramps are built on.
    """

    shifts: Float[Array, "n 2"]
    log_weights: Float[Array, " n"]
    box_size: int

    def ramps(self) -> Complex[Array, "n y x"]:
        """Phase ramps that translate a Fourier image: `fft2(image) * ramps()[j]` shifts it by `shifts[j]`.

        Returns:
            The ramps, on the unshifted fft grid.
        """
        return phase_ramp(self.shifts, self.box_size)


@eqx.filter_jit
def geodesic_error(angles: Float[Array, "n 3"], reference: Float[Array, "n 3"]) -> Float[Array, " n"]:
    """Angle of the rotation taking one pose onto another, in degrees.

    From the chordal distance, `theta = 2 arcsin(||Ra - Rb||_F / (2 sqrt 2))`, not
    `arccos((tr - 1) / 2)`, which has a noise floor near the identity in float32. Jitted so the vmap
    is not dispatched op by op.

    Args:
        angles: (n, 3) Euler angles in degrees.
        reference: (n, 3) poses to compare against.

    Returns:
        The geodesic distance on SO(3), in `[0, 180]`.

    Rationale: docs/rationale/poses.md
    """
    if angles.shape != reference.shape:
        raise ValueError(f"pose shapes differ: {angles.shape} vs {reference.shape}")
    matrices = jax.vmap(lambda row: rotation_of(row).as_matrix())
    difference = matrices(angles) - matrices(reference)
    chord = jnp.sqrt(jnp.einsum("nij,nij->n", difference, difference))
    return 2.0 * jnp.degrees(jnp.arcsin(jnp.clip(chord / (2.0 * jnp.sqrt(2.0)), 0.0, 1.0)))


def rotation_of(angles: Float[Array, " 3"]) -> SO3:
    """The rotation of one Euler triple, which is where the angle convention is pinned.

    Args:
        angles: (phi, theta, psi) in degrees, cryoJAX intrinsic zyz.

    Returns:
        The rotation.
    """
    return cxs.EulerAnglePose(phi_angle=angles[0], theta_angle=angles[1], psi_angle=angles[2]).rotation


def euler_from_relion(euler: Float[np.ndarray, "n 3"]) -> Float[np.ndarray, "n 3"]:
    """RELION's `(rot, tilt, psi)` as the triple `rotation_of` renders at.

    Both conventions are intrinsic ZYZ in degrees, so the mapping looks like the identity but is not:
    it negates all three angles in place, keeping the order. That is NOT the inverse rotation, which is
    the reversed negation `(-psi, -tilt, -rot)`. Valid only where the templates are in the SAME frame
    the poses refer to and the images have been recentered by `metadata.recenter` at `sign=+1`.

    Args:
        euler: `(n, 3)` of `(rot, tilt, psi)` in degrees, as a RELION star spells them.

    Returns:
        `(n, 3)` in degrees for `rotation_of` and for `CryoJAXModel.bare_contrast`.

    Rationale: docs/rationale/poses.md
    """
    return -np.asarray(euler, float)


def balanced_psi(n_views: int) -> int:
    """The `n_psi` matching S1 spacing to S2 spacing, rounded to the NEAREST multiple of four.

    Equating S2 spacing `sqrt(4 pi / n_views)` to S1 spacing `2 pi / n_psi` gives
    `n_psi = sqrt(pi * n_views)`. A multiple of four buys the quarter-turn symmetry; nearest rather
    than up, since rounding up costs the bank its isotropy and covering.

    Args:
        n_views: Points on the sphere.

    Returns:
        The matching `n_psi`, a multiple of four and at least four.

    Raises:
        ValueError: If `n_views` is not positive.

    Rationale: docs/rationale/poses.md
    """
    if n_views < 1:
        raise ValueError(f"n_views must be positive, got {n_views}")
    ideal = float(np.sqrt(np.pi * n_views))
    return max(4, int(4 * round(ideal / 4.0)))


def shift_grid(box_size: int, max_shift: float, sigma: float | None = None, step: float = 1.0) -> ShiftGrid:
    """Square grid of candidate translations within `max_shift` pixels of the origin.

    Args:
        box_size: Image size in pixels.
        max_shift: Half-width of the window in pixels.
        sigma: Width of an isotropic Gaussian prior in pixels. `None` gives a uniform prior.
        step: Grid spacing in pixels. The candidate count, and so the cost, is quadratic in `1/step`.

    Returns:
        A grid of `(2 * floor(max_shift / step) + 1)**2` candidates.
    """
    if step <= 0.0 or max_shift < 0.0:
        raise ValueError(f"need step > 0 and max_shift >= 0, got step={step}, max_shift={max_shift}")

    reach = int(np.floor(max_shift / step))
    axis = step * np.arange(-reach, reach + 1)
    shift_y, shift_x = np.meshgrid(axis, axis, indexing="ij")
    shifts = np.stack([shift_y.ravel(), shift_x.ravel()], axis=-1)
    radius_squared = (shifts**2).sum(-1)
    log_weights = np.zeros_like(radius_squared) if sigma is None else -radius_squared / (2.0 * sigma**2)
    return ShiftGrid(
        shifts=jnp.asarray(shifts),
        log_weights=jnp.asarray(log_weights - logsumexp(log_weights)),
        box_size=box_size,
    )


def mean_square_miss(grid: PoseGrid, n_probe: int = 4000, seed: int = 0) -> float:
    """`E|delta|^2` for this bank: how far a Haar-drawn rotation sits from its nearest node, squared.

    The geometric half of the finite-bank deficit. A property of the bank alone. Estimated by
    sampling, with relative standard error near `1/sqrt(n_probe)`.

    Args:
        grid: The bank. Its weights are ignored -- a nearest-node distance has nothing to do with
            quadrature weight, which is why this is not a method on `PoseGrid`.
        n_probe: Haar rotations to draw.
        seed: Draw seed, fixed by default so repeated calls agree.

    Returns:
        The mean squared geodesic miss, in radians squared.
    """
    probe = Rotation.random(n_probe, random_state=seed)
    nodes = Rotation.from_euler("zyz", np.asarray(grid.angles, dtype=np.float64), degrees=True)
    # |q . q'| because q and -q are the same rotation; the argmax is the nearest node
    nearest = np.abs(probe.as_quat() @ nodes.as_quat().T).argmax(axis=1)
    residual = (probe * nodes[nearest].inv()).as_rotvec()
    return float(np.mean(np.sum(residual**2, axis=1)))


_USABLE_TRIANGULATIONS = tuple(f * f for f in range(1, 41))
"""Triangulation numbers a bank may be built at: `T = f^2`, Class I only.

Class I and II tie on covering, so the restriction rests on quantization: Class I's counts are spaced
`(f+1)^2/f^2 = 1 + 2/f`, dense in log exactly where a rung is expensive, while Class II's extra sizes
fall at the coarse end where quantization is already harmless. `subdivision_of` still knows Class II,
which is how Class III is identified and refused; this constant is the policy, that function the math.

A TUPLE, ascending: `_choose_bank`'s early `break` depends on the order, and immutability keeps the
`@cache`d banks consistent -- call `_choose_bank.cache_clear()` to patch it in a test.
"""


def sphere_angles(n_nodes: int) -> Float[Array, "n 3"]:
    """Viewing directions nearest `n_nodes` in count, in-plane angle left at zero.

    The bank `rungs.refine` searches: it maximizes over these and treats `psi` continuously, so only
    how evenly the DIRECTIONS cover matters and quadrature weights are meaningless. A geodesic sphere
    (a spherical 5-design, antipodally closed). The count is quantized to `10 * T + 2` over
    `_USABLE_TRIANGULATIONS`, nearest in log, so read the returned length rather than assume it.

    Args:
        n_nodes: Roughly how many directions to place.

    Returns:
        `(n, 3)` Euler triples in degrees, every one with `psi = 0`.

    Raises:
        ValueError: If `n_nodes` is not positive.
    """
    if n_nodes < 1:
        raise ValueError(f"the number of directions must be positive, got {n_nodes}")
    directions = geodesic_directions(min(_USABLE_TRIANGULATIONS, key=lambda t: abs(np.log((10.0 * t + 2.0) / n_nodes))))
    return jnp.asarray(
        np.stack([directions[:, 0], directions[:, 1], np.zeros(len(directions))], axis=-1).astype(np.float32)
    )


_ICOSAHEDRON = np.array(
    [[0.0, s1, s2 * (1.0 + 5.0**0.5) / 2.0] for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
    + [[s1, s2 * (1.0 + 5.0**0.5) / 2.0, 0.0] for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
    + [[s2 * (1.0 + 5.0**0.5) / 2.0, 0.0, s1] for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
)
"""The twelve icosahedral vertices, as cyclic permutations of `(0, +/-1, +/-phi)`, un-normalized."""


def subdivision_of(triangulation: int) -> tuple[int, int]:
    """The `(m, n)` Goldberg vector for a triangulation number, or a refusal.

    `T = m^2 + m n + n^2`, and two of the three classes name a sphere this module can build:

        Class I    `(f, 0)`, `T = f^2`   -- 1, 4, 9, 16, 25, ...
        Class II   `(f, f)`, `T = 3 f^2` -- 3, 12, 27, 48, 75, ...
        Class III  `(m, n)` otherwise    -- 7, 13, 19, 21, ... REFUSED

    Class II constructs but is not SELECTED (`_USABLE_TRIANGULATIONS` is Class I alone); identifying it
    here is how Class III is told apart. Class III is chiral and covers worse, so it is refused.

    Args:
        triangulation: `T`. Must be a square or three times a square.

    Returns:
        `(m, n)`.

    Raises:
        ValueError: If `T` is not positive, or is a Class III triangulation number.

    Rationale: docs/rationale/poses.md
    """
    if triangulation < 1:
        raise ValueError(f"the triangulation number must be positive, got {triangulation}")
    root = round(triangulation**0.5)
    if root * root == triangulation:
        return root, 0
    third = triangulation / 3.0
    root = round(third**0.5)
    if triangulation % 3 == 0 and root * root * 3 == triangulation:
        return root, root
    # a T with no (m, n) at all is a different mistake from a chiral one, and saying "Class III"
    # sends a caller who typed `hopf_grid(2)` looking for a chirality problem that does not exist
    chiral = any(m * m + m * n + n * n == triangulation for m in range(1, triangulation) for n in range(1, m))
    if chiral:
        raise ValueError(
            f"T = {triangulation} is a Class III triangulation number, which is chiral: it keeps the "
            "icosahedral rotations and drops the inversion, so the antipodal half of the 8x expansion "
            "has nothing to act on. Use a square (Class I: 1, 4, 9, 16, ...) or three times a square "
            "(Class II: 3, 12, 27, 48, ...)."
        )
    raise ValueError(
        f"T = {triangulation} is not a triangulation number at all -- there is no (m, n) with "
        "m^2 + mn + n^2 = T, so it names no geodesic sphere. Use a square (Class I: 1, 4, 9, 16, ...) "
        "or three times a square (Class II: 3, 12, 27, 48, ...)."
    )


_SPHERICAL_AVERAGE_STEPS = 32
"""Iteration ceiling for the Buss-Fillmore fixed point in `geodesic_directions`.

It converges quadratically and the loop breaks when the tangent step falls below 1e-14, so this is a
ceiling, not a count. Fixed rather than tolerance-only so the construction is deterministic to the bit.
"""


def geodesic_directions(triangulation: int) -> Float[np.ndarray, "n 2"]:
    """Geodesic-sphere vertices as `(phi, theta)` in degrees: `10 * T + 2` of them.

    Subdivision of the icosahedron at triangulation number `T` (Class I or II; see `subdivision_of`),
    placed by the SPHERICAL barycentric average rather than interpolating in the plane and projecting.
    Every icosahedral orbit is a spherical 5-design, so an unweighted mean over it integrates Haar
    exactly to degree five (`<z> = 0`, `<z^2> = 1/3`, `<z^4> = 1/5`); that is what lets the bank carry
    UNIFORM weights.

    Args:
        triangulation: `T`, a square (Class I) or three times a square (Class II).

    Returns:
        `(n, 2)` of `(phi, theta)` in degrees, `theta` from the north pole, in a deterministic order.

    Raises:
        ValueError: If `T` is not positive or is Class III.

    Rationale: docs/rationale/poses.md
    """
    m, n = subdivision_of(triangulation)
    base = _ICOSAHEDRON / np.linalg.norm(_ICOSAHEDRON, axis=1, keepdims=True)
    corners = base[ConvexHull(base).simplices]

    # the face lattice: points i*(m, n) + j*(-n, m + n) inside the triangle (Class I, n = 0, is the
    # plain barycentric grid). Span is O(sqrt(T)) per axis: i and j leave the triangle by about m + n.
    span = 2 * (m + n) + 1
    index = np.arange(-span, span + 1)
    i_grid, j_grid = np.meshgrid(index, index, indexing="ij")
    x = (i_grid * m - j_grid * n) / triangulation
    y = (i_grid * n + j_grid * (m + n)) / triangulation
    keep = (x >= -1e-9) & (y >= -1e-9) & (x + y <= 1 + 1e-9)
    weights = np.stack([1.0 - x[keep] - y[keep], x[keep], y[keep]], axis=1)

    # Buss-Fillmore: the weighted spherical average is the point whose weighted log map sums to zero,
    # reached by walking that sum from the linear-then-project seed.
    point = np.einsum("mk,fkc->fmc", weights, corners)
    point /= np.linalg.norm(point, axis=-1, keepdims=True)
    for _ in range(_SPHERICAL_AVERAGE_STEPS):
        overlap = np.clip(np.einsum("fmc,fkc->fmk", point, corners), -1.0, 1.0)
        tangent = corners[:, None, :, :] - overlap[..., None] * point[:, :, None, :]
        length = np.linalg.norm(tangent, axis=-1, keepdims=True)
        logs = np.where(length > 1e-12, np.arccos(overlap)[..., None] * tangent / np.maximum(length, 1e-12), 0.0)
        step = np.einsum("mk,fmkc->fmc", weights, logs)
        size = np.linalg.norm(step, axis=-1, keepdims=True)
        if size.max() < 1e-14:
            break
        point = point * np.cos(size) + np.where(size > 0.0, step / np.maximum(size, 1e-300), 0.0) * np.sin(size)

    unit = point.reshape(-1, 3)
    unit /= np.linalg.norm(unit, axis=-1, keepdims=True)
    # rounded before de-duplicating because the same vertex is reached from up to six faces by
    # different arithmetic; nine places is far below the spacing and far above the discrepancy
    unit = np.unique(np.round(unit, 9), axis=0)
    theta = np.degrees(np.arccos(np.clip(unit[:, 2], -1.0, 1.0)))
    phi = np.degrees(np.arctan2(unit[:, 1], unit[:, 0])) % 360.0
    return np.stack([phi, theta], axis=1)


def hopf_grid(triangulation: int, n_psi: int | None = None) -> PoseGrid:
    """A Hopf-fibration bank on a geodesic sphere: a 5-design crossed with a uniform `psi`.

    Yershova, Jain, LaValle & Mitchell (IJRR 29:801, 2010); `balanced_psi` is their isotropy condition
    Eq. (9). The sphere is a subdivided icosahedron for exact antipodal closure and Haar integration
    to degree five (see `geodesic_directions`). Because every node's antipode is a node,
    `n_views * n_psi / 8` renders generate the bank, the rest by quarter turns and a flip.

    Args:
        triangulation: `T`, giving `10 * T + 2` viewing directions. Class I or II; see `subdivision_of`.
        n_psi: In-plane angles. `None` takes `balanced_psi(10 * T + 2)`. A multiple of four is required.

    Returns:
        The bank, `(10 * T + 2) * n_psi` nodes, uniform weights, views-outer and `psi`-inner;
        `expansion_plan` finds the antipodal partners itself rather than assuming an order.

    Raises:
        ValueError: If `T` is not usable or `n_psi` not a positive multiple of four.

    Rationale: docs/rationale/poses.md
    """
    directions = geodesic_directions(triangulation)
    n_views = len(directions)
    if n_psi is None:
        n_psi = balanced_psi(n_views)
    if n_psi < 4 or n_psi % 4:
        raise ValueError(
            f"n_psi must be a positive multiple of four, got {n_psi}. The quarter turns are what make "
            "three of every four in-plane angles free."
        )
    psi = np.arange(n_psi) * (360.0 / n_psi)
    angles = np.stack(
        [np.repeat(directions[:, 0], n_psi), np.repeat(directions[:, 1], n_psi), np.tile(psi, n_views)],
        axis=1,
    )
    n = n_views * n_psi
    return PoseGrid(angles=jnp.asarray(angles), log_weights=jnp.full(n, -float(np.log(n))))


_BALANCE_BAND = (0.8, 1.2)
"""How far `n_psi` may sit from the isotropic ideal `sqrt(pi * n_views)` before `pose_bank` refuses it.

The band is where the bank stays near-isotropic and still beats a spiral of equal node count. Outside
it the loss is severe and silent: too many `psi` resolves the in-plane angle far finer than the
viewing direction, spending templates to cover worse.
"""


@cache
def _choose_bank(n_nodes: int) -> tuple[int, int]:
    """The `(triangulation, n_psi)` whose node count lands nearest `n_nodes`.

    THE quantization rule, in one place, so `pose_bank` and everything keyed on bank size agree on
    which bank a request builds. The triangulations searched are `_USABLE_TRIANGULATIONS`, Class I.

    Args:
        n_nodes: The requested number of orientations.

    Returns:
        `(triangulation, n_psi)`: `hopf_grid(*result)` is the bank and `10 * result[0] + 2` its
        direction count.

    Raises:
        ValueError: If `n_nodes` is not positive.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    best: tuple[float, int, int] | None = None  # (cost, triangulation, n_psi)
    for triangulation in _USABLE_TRIANGULATIONS:
        n_views = 10 * triangulation + 2
        # stop once even the smallest legal `n_psi` overshoots by 4x, but never before the smallest
        # sphere has been considered: the smallest bank is 96 nodes and a caller may ask for fewer
        if best is not None and n_views * 4 > 4 * n_nodes:
            break
        ideal = float(np.sqrt(np.pi * n_views))
        # the balanced choice is always admissible: on a small sphere the band can contain no
        # multiple of four at all, and refusing there would leave the coarse end of a ladder unreachable
        allowed = {balanced_psi(n_views)} | {
            n for n in range(4, 4 * int(2.0 * ideal / 4.0) + 5, 4) if _BALANCE_BAND[0] <= n / ideal <= _BALANCE_BAND[1]
        }
        for n_psi in allowed:
            cost = abs(np.log(n_views * n_psi / n_nodes))
            if best is None or cost < best[0]:
                best = (cost, triangulation, n_psi)
    assert best is not None
    return best[1], best[2]


def pose_bank(n_nodes: int) -> PoseGrid:
    """The bank nearest `n_nodes`, chosen over `(triangulation, n_psi)` and kept near isotropic balance.

    THE constructor. A bank's node count is `(10 T + 2) * n_psi` rather than free, so this picks the
    pair whose product is closest in log. Quantization is coarse at the bottom, so two small requests
    can return the SAME bank. Read `grid.angles.shape[0]` back rather than assuming the count asked for.

    Args:
        n_nodes: Roughly how many templates to spend. Any positive integer.

    Returns:
        The bank. Uniform weights, `psi` innermost, antipodally closed -- so `expansion_plan` can
        rebuild it from an eighth of its nodes.

    Raises:
        ValueError: If `n_nodes` is not positive.

    Rationale: docs/rationale/poses.md
    """
    return hopf_grid(*_choose_bank(n_nodes))


@cache
def realized_nodes(requested: int) -> int:
    """How many orientations a `--nodes` request actually integrates over.

    `pose_bank` returns the nearest bank it can build, so few requests survive unchanged and a record
    stores only the request. A reader keying a curve by requested size would plot sizes never built.
    Cached, since a ladder glob asks the same sizes many times.

    Args:
        requested: The value passed to `--nodes`, as a filename or `arguments` block records it.

    Returns:
        `pose_bank(requested).angles.shape[0]`.
    """
    return int(pose_bank(requested).angles.shape[0])


def median_miss(angles: Float[np.ndarray, "n 3"], grid: PoseGrid) -> float:
    """Median geodesic distance, in degrees, from each of `angles` to the nearest node of `grid`.

    What a bank offers a GIVEN set of poses, as opposed to a Haar draw (`mean_square_miss`). The two
    come apart when the poses were themselves drawn from a bank -- a trap `chain._extrapolate_marginal`
    refuses a ladder on.

    Args:
        angles: `(n, 3)` Euler triples in degrees, the poses the particles were actually rendered at.
        grid: The bank.

    Returns:
        The median miss in degrees. Zero when every pose sits on a node.

    Raises:
        ValueError: If either input is empty.
    """
    pool = np.atleast_2d(np.asarray(angles, dtype=np.float64))
    nodes = np.asarray(grid.angles, dtype=np.float64)
    if pool.size == 0 or nodes.size == 0:
        raise ValueError("both the poses and the bank must be non-empty")
    a = Rotation.from_euler("zyz", pool, degrees=True).as_quat()
    b = Rotation.from_euler("zyz", nodes, degrees=True).as_quat()
    # |q . q'| because q and -q are the same rotation; the max over nodes is the nearest one
    overlap = np.abs(a @ b.T).max(axis=1)
    return float(np.median(2.0 * np.degrees(np.arccos(np.clip(overlap, 0.0, 1.0)))))


_PSI_DRIFT_TOLERANCE = 1e-3
"""Degrees a psi cycle may sit off `k * 360/n_psi` before it is refused as phase-offset.

Sized against float32, not against zero: a `PoseGrid` stores angles in float32, so a cycle whose
`n_psi` does not divide 360 cannot land on the lattice exactly, and at 360 degrees float32 resolution
is ~4e-5 deg. The tolerance sits above that and far below the smallest offset that breaks the mirror,
so it refuses arithmetic noise, not a wrong grid.
"""


def expansion_plan(grid: PoseGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Which nodes to render, and how each of the rest is built from one of them.

    The antipodal partners are FOUND, not assumed from an ordering: `geodesic_directions` sorts its
    vertices lexicographically, so a plan assuming a hemisphere-then-mirror layout would pair the
    wrong directions while still returning a plausible bank.

    Args:
        grid: A bank whose viewing directions are antipodally closed and whose `psi` values are a
            uniform cycle of length divisible by four -- `hopf_grid` builds one.

    Returns:
        `(rendered, source, turns, mirror)`. `rendered` indexes the `n / 8` nodes actually rendered.
        For every node `i`, `source[i]` is its position in the rendered set, `mirror[i]` whether to
        flip first, and `turns[i]` the quarter turns AFTER the flip -- already reversed where the flip
        demands it, so a caller needs to know nothing about handedness.

    Raises:
        ValueError: If the directions are not antipodally closed, or `psi` is not a uniform
            multiple-of-four cycle. A bank that half-satisfies the symmetry yields one that is half wrong.
    """
    angles = np.asarray(grid.angles, dtype=np.float64)
    view_rows, first = np.unique(angles[:, :2], axis=0, return_index=True)
    order = np.argsort(first)
    views = view_rows[order]
    n_views = len(views)
    n_psi = len(angles) // n_views
    if n_psi < 4 or n_psi % 4 or n_views % 2:
        raise ValueError(f"need an even view count and a multiple-of-four psi cycle, got {n_views} and {n_psi}")

    phi, theta = np.radians(views[:, 0]), np.radians(views[:, 1])
    normal = np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=1)
    partner = (normal @ -normal.T).argmax(axis=1)
    if not np.allclose((normal * normal[partner]).sum(axis=1), -1.0, atol=1e-6):
        raise ValueError("the viewing directions are not antipodally closed, so the mirror buys nothing")

    # the primary of each antipodal pair is whichever has the lower index, so the choice is
    # deterministic and independent of how the sphere happened to be listed
    is_primary = np.arange(n_views) < partner

    # A mirrored node assumes its partner sits at `phi + 180`, exact away from the poles. At a POLE
    # phi is free (both stored at zero), so the missing 180 is an image-plane half turn `extra` fixes.
    offset = np.remainder(views[:, 0] + 180.0 - views[partner, 0] + 180.0, 360.0) - 180.0
    extra = np.rint(offset / 90.0).astype(int) % 4
    if not np.allclose(np.remainder(extra * 90.0 - offset + 180.0, 360.0) - 180.0, 0.0, atol=1e-3):
        raise ValueError(f"a pole is stored at a phi that is not a quarter turn from its partner: {offset}")

    quarter = n_psi // 4
    view_of = {tuple(np.round(row, 6)): index for index, row in enumerate(views)}
    node_view = np.array([view_of[tuple(np.round(row, 6))] for row in angles[:, :2]])
    node_psi = np.rint(angles[:, 2] / (360.0 / n_psi)).astype(int) % n_psi
    # Check the psi cycle's VALUES, not just its count. The mirror negates the psi ANGLE while the code
    # negates the psi INDEX, agreeing only for a cycle at zero, so an offset leaves the flipped half wrong.
    step = 360.0 / n_psi
    drift = np.remainder(angles[:, 2] - node_psi * step + 180.0, 360.0) - 180.0
    if not np.allclose(drift, 0.0, atol=_PSI_DRIFT_TOLERANCE):
        raise ValueError(
            f"psi is not the cycle k * {step} deg; it drifts by up to {np.abs(drift).max():.4f} deg. "
            "The mirror negates the psi angle and this negates the psi index, which agree only for a "
            "cycle starting at zero, so an offset one leaves the flipped half of the bank unreachable."
        )
    # Then COVERAGE, which the offset check cannot see: {0, 90, 90, 270} drifts nowhere yet collides
    # two nodes onto one slot. The `bincount` asks for exactly one node per (view, psi) cell.
    counts = np.bincount(node_view * n_psi + node_psi, minlength=n_views * n_psi)
    if not np.array_equal(counts, np.ones_like(counts)):
        bad = int(np.argmax(counts != 1)) // n_psi
        raise ValueError(
            f"view {bad} does not carry the psi cycle 0..{n_psi - 1} exactly once: "
            f"got {sorted(node_psi[node_view == bad].tolist())}"
        )

    rendered: list[int] = []
    source = np.empty(len(angles), dtype=int)
    turns, mirror = np.empty(len(angles), dtype=int), np.zeros(len(angles), dtype=bool)
    slot_of: dict[tuple[int, int], int] = {}
    for primary in np.flatnonzero(is_primary):
        for p in range(quarter):
            slot_of[(int(primary), p)] = len(rendered)
            rendered.append(int(np.flatnonzero((node_view == primary) & (node_psi == p))[0]))

    for node in range(len(angles)):
        view, p = int(node_view[node]), int(node_psi[node])
        flipped = not is_primary[view]
        base_view = int(partner[view]) if flipped else view
        source_p = (-p) % n_psi if flipped else p
        source[node] = slot_of[(base_view, source_p % quarter)]
        spin = source_p // quarter
        turns[node] = (4 - spin + extra[view]) % 4 if flipped else spin
        mirror[node] = flipped
    return np.asarray(rendered, dtype=int), source, turns, mirror


def expansion_plan_or_none(grid: PoseGrid, box_size: int) -> tuple | None:
    """`expansion_plan` for a bank that has one, and `None` for a bank that does not.

    Refusing is the plan's job, for structural reasons (directions not antipodally closed, a `psi`
    cycle not a multiple of four); the right answer for a caller's own grid without a plan is the whole
    bank rendered, not an exception. An ODD box is refused here too. The cost of being wrong is only
    speed, never the number.

    Args:
        grid: The bank about to be scored.
        box_size: Pixels per side of the templates it will be scored against.

    Returns:
        `(rendered, source, turns, mirror)`, or `None`.
    """
    if box_size % 2:
        return None
    try:
        return expansion_plan(grid)
    except ValueError:
        return None
