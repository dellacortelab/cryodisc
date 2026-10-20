"""The optics as they reached each particle: one CTF per defocus group, gathered per particle.

A description of the instrument, not a step of any fit -- it reads only a plan, a defocus and a box.
The grouping makes a per-particle transfer affordable; how many groups is not free but derived by
`defocus_groups_for`, since too few bins is not a coarser answer but a wrong one.
"""

from typing import Self

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Complex, Float, Int

from cryodisc.grids.fourier import frequency_grid
from cryodisc.microscope.ctf import ctf, wavelength
from cryodisc.microscope.plan import CollectionPlan

_GROUP_PHASE_TOLERANCE = 0.25 * jnp.pi
"""Most CTF phase, in radians at Nyquist, that a defocus group may span internally.

A quarter turn. Past about a half the group's CTF has the wrong SIGN over part of the band for part
of its members, and templates anti-correlate with the data there.
"""


def defocus_groups_for(plan: CollectionPlan, tolerance: float = float(_GROUP_PHASE_TOLERANCE)) -> int:
    """How many defocus bins a plan needs before the binning stops costing anything.

    Within a bin of width `w` the defocus term of `chi` spans `pi lambda w k^2`, so the bin count
    grows with the defocus range and with the SQUARE of the resolution. Too few bins understates a
    known-pose `d^2` with no sign that anything is wrong.

    Args:
        plan: The collection, supplying the defocus range, the voltage and the Nyquist.
        tolerance: Phase span allowed per group, in radians at Nyquist.

    Returns:
        The bin count. At least one, and `Transfer.from_defocus` caps it at the particle count.
    """
    span = plan.defocus_range[1] - plan.defocus_range[0]
    phase = jnp.pi * wavelength(plan.voltage_in_kv) * span * plan.nyquist**2
    return max(1, int(jnp.ceil(phase / tolerance)))


class Transfer(eqx.Module):
    """A per-group transfer function and the group each particle belongs to.

    The two travel together because neither means anything alone. Grouping by defocus is also what
    makes a per-particle transfer affordable: the fit only ever needs sums over a group.

    Args:
        values: Transfer per group on the unshifted fft grid, real, (g, y, x).
        index: Group of each particle, (n,), each in `[0, g)`.
    """

    values: Float[Array, "g y x"]
    index: Int[Array, " n"]

    @classmethod
    def from_defocus(
        cls, defocus: Float[Array, " n"], plan: CollectionPlan, box_size: int, n_groups: int | None = None
    ) -> Self:
        """Pool a per-particle defocus into `n_groups` bins and give each its CTF.

        Equal-count bins rather than equal-width, so no group is empty and none carries the whole
        stack; each group's CTF is evaluated at the mean defocus of the particles in it. This is the
        builder `ParticleStack.defocus` is named for.

        Args:
            defocus: Underfocus of each particle in A, (n,), as `particles.sim.ParticleStack` returns.
            plan: The collection the CTF is evaluated under.
            box_size: Image size in pixels.
            n_groups: Number of bins, capped at the particle count. `None` asks
                `defocus_groups_for` for enough of them, which is the only safe default: too few is
                not a coarser answer but a wrong one. One group per particle is exact and is the
                expensive path this grouping exists to avoid.

        Returns:
            The transfer and each particle's group.

        Raises:
            ValueError: If `defocus` is empty or `n_groups` is not positive.
        """
        if n_groups is None:
            n_groups = defocus_groups_for(plan)
        if defocus.size < 1 or n_groups < 1:
            raise ValueError(f"need at least one particle and one group, got {defocus.size} and {n_groups}")
        # capped: an empty bin averages 0/0 into an all-NaN CTF row that `fit_noise_model` then
        # spreads to the whole fit via `0 * nan`, silently
        n_groups = min(n_groups, defocus.size)
        # int64: `arange(n) * n_groups` overflows int32 past 2^31 and wraps the index negative
        rank = np.arange(defocus.size, dtype=np.int64) * n_groups // defocus.size
        order = jnp.argsort(defocus)
        index = jnp.zeros_like(order).at[order].set(jnp.asarray(rank))
        centers = jnp.zeros(n_groups).at[index].add(defocus) / jnp.zeros(n_groups).at[index].add(1.0)
        values = ctf(frequency_grid(box_size, plan.pixel_size), defocus_in_angstroms=centers[:, None, None], plan=plan)
        return cls(values=values, index=index.astype(jnp.int32))

    def checked_index(self) -> Int[Array, " n"]:
        """`index`, guarded against the silent gather JAX would otherwise do.

        The group ids are traced, so nothing catches them at trace time, and an out-of-bounds gather
        CLAMPS rather than raising: 1-based ids would put every particle on a neighboring group's
        transfer and the fit would converge, to the wrong answer. Every consumer gathers through
        this rather than through `index`.

        Returns:
            The index, unchanged.
        """
        return eqx.error_if(
            self.index,
            (self.index < 0) | (self.index >= self.values.shape[0]),
            f"transfer.index out of range: group ids are 0-based and must be below {self.values.shape[0]}",
        )

    def for_particle_count(self, n_particles: int) -> Self:
        """The same optics, assigned to a different number of particles.

        What a null or planted pool needs: it has no defocus of its own but must be scored through the
        transfer the MEASUREMENT used. The index is CYCLED rather than collapsed, so the drawn pool
        inherits the measurement's own mix of groups, deterministically and without a key.

        Args:
            n_particles: How many particles to assign. May be larger or smaller than the original.

        Returns:
            A transfer with the same `values` and an index of length `n_particles`.

        Raises:
            ValueError: If `n_particles` is not positive.
        """
        if n_particles < 1:
            raise ValueError(f"need at least one particle, got {n_particles}")
        return type(self)(values=self.values, index=self.index[jnp.arange(n_particles) % self.index.size])


def fold_transfer(
    data: Complex[Array, "n k"],
    transfer: Transfer | None,
    box: tuple[int, int],
    *,
    shared_as_ones: bool = False,
) -> tuple[Complex[Array, "n k"], Float[Array, "g k"] | None, Array | None]:
    """The per-group CTF folded onto the data, with the squared copy a model power needs.

    `C` is REAL, so `Re(Y conj(C M)) == Re((C Y) conj(M))`: the transfer folds onto the fixed DATA
    once rather than every iteration, leaving a shared pose bank untouched, which is why one bank can
    serve particles at different defocus. Shared by `particles.em` and `rungs.marginal` so they validate
    a transfer identically.

    Args:
        data: The stack, flattened to (n, k).
        transfer: Per-group transfer, or `None` for a shared one.
        box: `(box_y, box_x)`, which the transfer's own values must match.
        shared_as_ones: Return a shared transfer as one group of ones rather than as `None`. What a
            caller wants when it gathers `squared[index]` unconditionally; the `None` form exists so
            a caller that would otherwise multiply the whole bank by ones can skip the multiply,
            which over a pose bank is the largest array in the problem.

    Returns:
        `(weighted, squared, index)`. `squared` and `index` are `None` for a shared transfer unless
        `shared_as_ones`, in which case they are a single group of ones and an all-zero index.

    Raises:
        ValueError: If the transfer carries the wrong number of indices or the wrong box.
    """
    n_particles, n_pixels = data.shape
    if transfer is None:
        if shared_as_ones:
            return data, jnp.ones((1, n_pixels)), jnp.zeros(n_particles, int)
        return data, None, None
    if transfer.index.shape != (n_particles,):
        raise ValueError(f"transfer.index must be one index per particle, got {transfer.index.shape}")
    if transfer.values.shape[1:] != box:
        raise ValueError(f"transfer shape {transfer.values.shape[1:]} does not match images {box}")
    values = transfer.values.reshape(-1, n_pixels)
    index = transfer.checked_index()
    return data * values[index], values**2, index
