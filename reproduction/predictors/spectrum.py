"""Where in spatial frequency the state difference lives, before any collection is chosen.

Returns `local_information`'s sum `|a (mu_A - mu_B)|^2 / sigma^2(k)` BINNED BY SHELL with the
whitening left off, so `d^2(plan) = sum_j S_j / sigma^2(k_j; plan)` for any plan sharing the OPTICS,
box and pixel size it was rendered under. Dose, the envelopes and a band limit are then arithmetic on
the stored shells; the CTF is not, since it is applied at render time and cannot be divided back out,
so sweeping defocus, voltage or pixel size costs a render pass each.

Rationale: docs/rationale/rungs/spectrum.md
"""

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import shell_indices, shell_sum, whitening_weight
from cryodisc.grids.poses import rotation_of
from cryodisc.microscope import CollectionPlan
from cryodisc.resolution.noise import noise_spectrum

_POSE_CHUNK = 24
"""Poses rendered at once.

Peak memory is linear in `chunk * n_atoms`, the separable kernel contracting over every atom; this
path renders one image per pose per state where the local rung renders seven.
"""


def shell_weight(plan: CollectionPlan, box_size: int, n_shells: int, *, rmsf: float) -> np.ndarray:
    """`1 / sigma^2` averaged within each shell, which is all a shell sum can see.

    Averaged over the modes of the shell rather than evaluated at its center, because the two differ
    where `sigma^2` curves and the average is what the full-grid sum actually applies.

    Args:
        plan: The collection.
        box_size: Image size in pixels.
        n_shells: Shells the spectrum was binned on.
        rmsf: Per-axis intra-state RMS fluctuation in A, the model's Debye-Waller knob. `B = 8 pi^2
            rmsf^2`, so this is where a specimen B factor enters.

    Returns:
        One weight per shell, zero where the shell holds no mode.
    """
    index, _ = shell_indices(box_size, n_shells)
    weight = np.asarray(whitening_weight(noise_spectrum(plan, box_size, rmsf=rmsf))).ravel()
    flat = np.asarray(index).ravel()
    total = np.bincount(flat, weights=weight, minlength=n_shells + 1)[:n_shells]
    counts = np.bincount(flat, minlength=n_shells + 1)[:n_shells]
    return np.where(counts > 0, total / np.maximum(counts, 1), 0.0)


class DifferenceSpectrum(eqx.Module):
    """The state difference's power per Fourier shell, and the shells it was binned on.

    Args:
        power: `S_j`, summed over the modes of shell `j` and averaged over poses, in the units the
            renderer works in. Bare -- no CTF, no envelope, no whitening -- so a plan can be applied
            afterwards. DC is excluded, matching `whitening_weight`.
        frequency: Shell center in cycles per pixel. Divide by the pixel size for inverse angstroms.
        n_poses: How many poses it was averaged over, since the mean is what makes it a pair
            property rather than a projection's.
    """

    power: Float[Array, " n_shells"]
    frequency: Float[Array, " n_shells"]
    n_poses: int


@eqx.filter_jit
def _binned(
    model_a: CryoJAXModel,
    model_b: CryoJAXModel,
    angles: Float[Array, "n 3"],
    index: Array,
    n_shells: int,
    amplitude: Float[Array, ""],
    ctf_params: CTFParams | None,
    chunk: int,
) -> Float[Array, " n_shells"]:
    """The shell sums, averaged over poses. Split out so the whole reduction is one jit."""

    def at_pose(pose_angles: Float[Array, " 3"]) -> Float[Array, " n_shells"]:
        pose = cxs.QuaternionPose(wxyz=rotation_of(pose_angles).wxyz)
        difference = amplitude * (
            jnp.fft.fft2(model_a.render_at_pose(pose, ctf_params))
            - jnp.fft.fft2(model_b.render_at_pose(pose, ctf_params))
        )
        # DC dropped before binning, not after: zeroing shell 0 instead would take the coarsest
        # signal with it. `whitening_weight` drops the same mode, so the sum reproduces `local_information`.
        squared = (jnp.abs(difference) ** 2).at[0, 0].set(0.0)
        return shell_sum(squared, index, n_shells)[:n_shells]

    # chunked for the same reason `local_information` chunks: the atom render contracts over every
    # atom at once, and peak memory is linear in the poses held
    return jax.lax.map(at_pose, angles, batch_size=min(chunk, angles.shape[0])).mean(axis=0)


def difference_spectrum(
    model_a: CryoJAXModel,
    model_b: CryoJAXModel,
    angles: Float[Array, "n 3"],
    *,
    box_size: int,
    n_shells: int | None = None,
    amplitude: float = 1.0,
    ctf_params: CTFParams | None = None,
    chunk: int = _POSE_CHUNK,
) -> DifferenceSpectrum:
    """The pose-averaged power of `mu_A - mu_B`, per Fourier shell.

    Args:
        model_a: Forward model of state A.
        model_b: Forward model of state B, on the same image config.
        angles: `(n, 3)` poses as Euler angles in degrees, cryoJAX intrinsic zyz. The average over
            them is what makes this a property of the pair rather than of one view.
        box_size: Image size in pixels, which sets the shell grid.
        n_shells: Shells out to the axis Nyquist. `None` gives `box_size // 2`, one per integer
            radius, which is as fine as the grid supports.
        amplitude: Template scale `a`. The power scales as its square.
        ctf_params: Optics applied at render time. **Pass the real optics.** `None` renders the
            bare integrated potential, which is a different quantity from contrast rather than an
            unscaled version of it, so a spectrum taken that way cannot be paired with
            `resolution.noise_spectrum` or given a transfer afterwards; it is a diagnostic.
        chunk: Poses rendered at once; see `_POSE_CHUNK`.

    Returns:
        The spectrum.

    Raises:
        ValueError: If the angles are not Euler triples.
    """
    if angles.shape[-1] != 3:
        raise ValueError(f"angles must be (n, 3) Euler triples, got {angles.shape}")
    shells = box_size // 2 if n_shells is None else n_shells
    index, frequency = shell_indices(box_size, shells)
    power = _binned(model_a, model_b, angles, index, shells, jnp.asarray(amplitude), ctf_params, chunk)
    return DifferenceSpectrum(power=power, frequency=frequency, n_poses=int(angles.shape[0]))


def d2_from_spectrum(
    spectrum: DifferenceSpectrum,
    weight: Float[Array, " n_shells"] | np.ndarray,
) -> float:
    """`d^2` implied by a spectrum under a per-shell weight `1 / sigma^2(k)`.

    The shell approximation: `sigma^2` varies WITHIN a shell, so this agrees with the full-grid sum
    only to the extent the shells are fine. The driver records both, so the gap is a measured number.

    Args:
        spectrum: As `difference_spectrum` returns it.
        weight: `1 / sigma^2` at each shell's frequency, including whatever transfer applies.

    Returns:
        The sum.
    """
    return float(jnp.sum(spectrum.power * jnp.asarray(weight)))


def half_power_spacing(spectrum: DifferenceSpectrum, weight: Float[Array, " n_shells"] | np.ndarray) -> float:
    """The spacing by which half of `d^2` has accumulated, in the units `frequency` is in.

    This is Fig 7's `s`, computed from the two structures and the collection before data exist.
    Interpolated within the crossing shell rather than snapped to it, since the shells are wide at the
    coarse end where most pairs cross.

    Args:
        spectrum: As `difference_spectrum` returns it.
        weight: Per-shell `1 / sigma^2`, as `d2_from_spectrum` takes it.

    Returns:
        `1 / k_half`, with `k_half` in the units of `spectrum.frequency`. Infinite when the
        weighted power is zero, which is a pair with no difference rather than a coarse one.
    """
    contribution = np.asarray(spectrum.power) * np.asarray(weight)
    total = contribution.sum()
    if total <= 0:
        return float("inf")
    cumulative = np.cumsum(contribution) / total
    crossing = int(np.searchsorted(cumulative, 0.5))
    frequency = np.asarray(spectrum.frequency)
    if crossing == 0:
        return float(1.0 / frequency[0])
    below, above = cumulative[crossing - 1], cumulative[crossing]
    span = above - below
    fraction = 0.0 if span <= 0 else (0.5 - below) / span
    return float(1.0 / (frequency[crossing - 1] + fraction * (frequency[crossing] - frequency[crossing - 1])))


def spacing_of_shells(frequency: Float[Array, " n_shells"] | np.ndarray, pixel_size: float) -> np.ndarray:
    """Shell frequencies in cycles per pixel, as spacings in angstroms.

    Args:
        frequency: Shell centers in cycles per pixel.
        pixel_size: Pixel size in angstroms.

    Returns:
        `pixel_size / k`, one per shell.
    """
    return pixel_size / np.asarray(frequency)


__all__ = [
    "DifferenceSpectrum",
    "d2_from_spectrum",
    "difference_spectrum",
    "half_power_spacing",
    "spacing_of_shells",
]
