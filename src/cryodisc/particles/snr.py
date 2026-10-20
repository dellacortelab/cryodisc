"""The whitened signal power a pool's whole ladder lives on, and its square root, the SNR.

    P = mean_pose sum_k a^2 |M(k)|^2 <T^2>(k) / sigma^2(k)

Every collection parameter reaches a rung only through P, so it is what an `OperatingPoint` records
and what two runs must agree on before their rungs are divided. `sqrt(P)` is the per-particle
matched-filter SNR on `resolution.predict_resolution`'s own definition and enters the particle count
quadratically. Nothing here fits: `em` supplies the contrast and spectrum, `metadata` the transfer.
"""

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, Complex, Float

from cryodisc.forward.expansion import expand_templates, pose_chunks
from cryodisc.grids.fourier import whitening_weight
from cryodisc.microscope import Transfer


def whitened_power(
    bank: Complex[Array, "p y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    transfer: Transfer | None = None,
    contrast: float = 1.0,
) -> float:
    """`mean_pose sum_k a^2 |M(k)|^2 <T^2>(k) / sigma^2(k)`, the exposure scalar of a pool.

    Averaged over the transfer's defocus GROUPS, exactly (whitened power is linear in `T^2`) and over
    PARTICLES, each group weighted by its population, so unequally populated groups report the same
    quantity as equal-count ones.

    Args:
        bank: fft2 of the templates, CTF-FREE. Passing a bank the transfer is already in counts it
            twice, giving roughly the square of the right number.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        transfer: The per-group transfer the pool is scored with. `None` treats it as unity.
        contrast: Template scale `a`, from `em.fit_noise_model`.

    Returns:
        The whitened signal power, DC dropped as everywhere else in this package.
    """
    weight = whitening_weight(noise_spectrum)
    if transfer is not None:
        # weighted per particle by each group's population: metadata.transfer_from_meta bins by phase
        # tolerance into very unequal groups, and em.py already averages per particle through [index]
        counts = jnp.bincount(transfer.checked_index(), length=transfer.values.shape[0]).astype(weight.dtype)
        # tensordot rather than broadcast-then-sum, which materializes a second (g, y, x) array
        weight = weight * jnp.tensordot(counts, transfer.values**2, axes=1) / jnp.maximum(counts.sum(), 1.0)
    return float(contrast**2 * jnp.mean(jnp.sum(jnp.abs(bank) ** 2 * weight, axis=(-2, -1))))


def streamed_whitened_power(
    render: "Callable[[Float[Array, 'c 3']], Complex[Array, 'c y x']]",
    angles: Float[Array, "p 3"],
    noise_spectrum: Float[Array, "y x"],
    *,
    transfer: Transfer | None = None,
    contrast: float = 1.0,
    expansion: tuple | None = None,
    pose_chunk: int = 8192,
) -> float:
    """`whitened_power` with the bank rendered in chunks, so `p` is not memory-bound.

    Exact, not approximate: `whitened_power` is a MEAN over the pose axis and composes as the
    count-weighted average over any partition, so the chunk size cannot move the answer. Takes the
    same `expansion` the scorer does, without which the SNR pass repeats a full render.

    Args:
        render: Maps `(c, 3)` angles to `(c, y, x)` fft2 CTF-FREE templates.
        angles: `(p, 3)` Euler angles of the bank, in degrees.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        transfer: The per-group transfer the pool is scored with, or `None` for unity.
        contrast: Template scale `a`.
        expansion: `poses.expansion_plan_or_none` for this bank, or `None` to render every node.
        pose_chunk: Poses per chunk. Affects only peak memory.

    Returns:
        The whitened signal power, identical to `whitened_power` on the whole bank.
    """
    total, seen = 0.0, 0
    for piece, nodes, plan in pose_chunks(angles, pose_chunk, expansion):
        templates = render(piece)
        if plan is not None:
            templates = expand_templates(templates, *plan)
        count = templates.shape[0]
        total += count * whitened_power(templates, noise_spectrum, transfer=transfer, contrast=contrast)
        seen += count
    if seen != angles.shape[0]:
        raise ValueError(f"the chunks covered {seen} of the bank's {angles.shape[0]} poses")
    return total / seen


def matched_filter_snr(
    bank: Complex[Array, "p y x"],
    noise_spectrum: Float[Array, "y x"],
    *,
    transfer: Transfer | None = None,
    contrast: float = 1.0,
) -> float:
    """Per-particle matched-filter SNR, on `resolution.predict_resolution`'s own definition.

    Args:
        bank: fft2 of the CTF-free templates.
        noise_spectrum: sigma^2(k) on the unshifted fft grid.
        transfer: The per-group transfer, or `None` for unity.
        contrast: Template scale `a`.

    Returns:
        `sqrt(whitened_power)`, the number `predict_resolution` takes as `snr` -- so running a
        calibration on a real pool and evaluating this inverts that definition exactly, rather than
        borrowing an SNR measured on some other dataset.
    """
    return float(jnp.sqrt(whitened_power(bank, noise_spectrum, transfer=transfer, contrast=contrast)))
