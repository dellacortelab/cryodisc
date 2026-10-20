"""Build seven of every eight templates from the eighth, by index permutation alone.

`poses.expansion_plan` says which nodes to render and how each of the rest is reached; this applies
that plan. The two identities below are pure index permutations in the Fourier plane -- exact for
ATOMIC templates, approximate for VOXEL ones whose interpolation is not exactly equivariant:

    psi + 90    a quarter turn about the beam axis, `rot90` plus a one-pixel roll
    antipode    the mirrored projection, a flip plus a one-pixel roll
"""

from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float

_MIRROR = lambda f: jnp.roll(jnp.flip(f, axis=-2), 1, axis=-2)
"""The antipodal view, in the Fourier plane: `flipud` in real space is a sign flip of the y frequency
index, and negating an fft index is a flip plus a roll of one, index zero staying put."""

_TURNS = (
    lambda f: f,
    lambda f: jnp.roll(jnp.rot90(f, k=-1, axes=(-2, -1)), 1, axis=-1),
    lambda f: jnp.roll(jnp.flip(jnp.flip(f, axis=-2), axis=-1), (1, 1), axis=(-2, -1)),
    lambda f: jnp.roll(jnp.flip(jnp.flip(jnp.rot90(f, k=-1, axes=(-2, -1)), axis=-2), axis=-1), 1, axis=-2),
)
"""`psi + 90 * k` for k = 0..3, in the Fourier plane.

Each is the Fourier counterpart of the real-space recipe `test_poses.py` pins against a real render:
`rot90(k=-1)` then roll `(0, 1)`, `rot90(k=2)` then roll `(1, 1)`, `rot90(k=1)` then roll `(1, 0)`.
Note the third is NOT `rot90` twice -- a 180-degree real rotation is a flip of both frequency indices,
which is why it appears here as two flips and a roll.
"""


def expand_templates(
    rendered: Complex[Array, "r y x"],
    source: np.ndarray,
    turns: np.ndarray,
    mirror: np.ndarray,
) -> Complex[Array, "n y x"]:
    """The whole bank, from the eighth of it that was rendered.

    Args:
        rendered: `fft2` of the templates at the nodes `expansion_plan` named, in its order.
        source: For each node of the bank, its position in `rendered`.
        turns: Quarter turns to apply to that source, AFTER the flip. Already reversed where the flip
            demands it, so no handedness is decided here.
        mirror: Whether that node's source must be flipped first.

    Returns:
        `(n, y, x)`, the bank in node order, `fft2` and unshifted, exactly as a render of every node
        would have produced it.

    Raises:
        ValueError: If the plan's three arrays disagree in length, index outside `rendered`, or the
            templates sit on an ODD box.
    """
    source, turns, mirror = np.asarray(source), np.asarray(turns), np.asarray(mirror)
    if not source.shape == turns.shape == mirror.shape:
        raise ValueError(f"the plan disagrees with itself: {source.shape}, {turns.shape}, {mirror.shape}")
    if source.size and (source.min() < 0 or source.max() >= rendered.shape[0]):
        raise ValueError(f"source indexes outside the {rendered.shape[0]} rendered templates")
    box_y, box_x = rendered.shape[-2:]
    if box_y % 2 or box_x % 2:
        # A refusal, not a correction: the quarter turn rotates about the box CENTER, and even vs odd
        # N put that center on different sample positions, needing different index permutations.
        raise ValueError(f"the expansion is an even-box identity and these templates are {box_y}x{box_x}")

    # The loop below writes each node only if its turn matches one of the four groups, so an invalid
    # turn would silently leave an all-zero row; these guards refuse it instead.
    if not np.issubdtype(turns.dtype, np.integer):
        raise ValueError(f"turns are quarter turns and must be integers, got dtype {turns.dtype}")
    if turns.size and not np.isin(turns, np.arange(len(_TURNS))).all():
        raise ValueError(f"turns are quarter turns and must lie in 0..3, got {turns.min()}..{turns.max()}")

    gathered = rendered[source]
    # the flip FIRST, then the turn: `expansion_plan` has already reversed the turn where the flip
    # demands it, so composing the other way round is wrong by the handedness it absorbed
    base = jnp.where(jnp.asarray(mirror)[:, None, None], _MIRROR(gathered), gathered)
    # `turns` is a numpy array, so the turn per node is known outside the trace; grouping applies each
    # turn to its own nodes once rather than building three throwaway templates per node.
    out = jnp.empty_like(base)
    for quarter_turn, turn in enumerate(_TURNS):
        taking = np.flatnonzero(turns == quarter_turn)
        if taking.size:
            out = out.at[taking].set(turn(base[taking]))
    return out


def pose_chunks(
    angles: Float[Array, "p 3"], step: int, expansion: tuple | None
) -> "Iterator[tuple[Float[Array, 'c 3'], np.ndarray | slice, tuple | None]]":
    """The bank, in pieces small enough to render and score one at a time.

    Two shapes. Without a plan a piece is a slice of the bank and every node in it is rendered. With
    one, the piece is a slice of the RENDERED EIGHTH, and the nodes it yields are the ones that eighth
    can build -- scattered through the bank rather than contiguous, because `geodesic_directions` sorts
    its vertices lexicographically. The slot chunk is an eighth of the pose chunk, so the expanded
    piece is the size the budget was written for.

    Args:
        angles: `(p, 3)` Euler angles of the whole bank, in degrees.
        step: Poses per chunk, as `_pose_chunk` sized it.
        expansion: `(rendered, source, turns, mirror)`, or `None` to render every node.

    Yields:
        `(piece, nodes, plan)`. `piece` is what to render; `nodes` indexes the bank rows those
        templates score; `plan` is the `expand_templates` arguments for this piece, or `None` when
        `piece` already IS the templates.

    Raises:
        ValueError: If the plan's arrays disagree with each other or with the bank.
    """
    n_angles = angles.shape[0]
    if expansion is None:
        for start in range(0, n_angles, step):
            yield angles[start : start + step], slice(start, start + step), None
        return

    rendered, source, turns, mirror = (np.asarray(part) for part in expansion)
    if not (len(source) == len(turns) == len(mirror) == n_angles):
        raise ValueError(f"the expansion plan describes {len(source)} nodes and the bank has {n_angles}")
    if source.size and (source.min() < 0 or source.max() >= len(rendered)):
        raise ValueError("the expansion plan sources a template outside the rendered set")

    slot_step = max(1, step // 8)
    for start in range(0, len(rendered), slot_step):
        stop = min(start + slot_step, len(rendered))
        nodes = np.flatnonzero((source >= start) & (source < stop))
        yield (
            angles[rendered[start:stop]],
            nodes,
            (source[nodes] - start, turns[nodes], mirror[nodes]),
        )
