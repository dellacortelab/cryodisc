"""The absolute scale: how much signal a particle carries, and how much noise an exposure buys.

    SSNR(k) = K N f S(k) <C_eff^2>(k) dose

`S(k)` is the whole particle's signal power (`signal_power`); `<C_eff^2>` is `transfer`'s; the noise
is white shot noise `sigma^2 = 1/dose`. `K` collects every absolute-scale factor common to all
targets into one constant, so the model is calibration-free up to a single number: `calibrate_scale`
on one target, `predict_resolution` on the next. The question is whether that `K` transfers.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
from cryojax.constants import PengScatteringFactorParameters
from jaxtyping import Array, Float, Int

from cryodisc.microscope import CollectionPlan
from cryodisc.resolution.transfer import effective_transfer_squared

FULL_DATASET_SSNR = 0.33
"""SSNR at the Rosenthal-Henderson FSC = 0.143 criterion, `FSC = SSNR / (SSNR + 1)` doubled to the
full dataset."""


def scattering_factor(atomic_numbers: Int[np.ndarray, " e"], k: Float[Array, " ..."]) -> Float[Array, "... e"]:
    """Peng electron scattering amplitude `f_e(k)`, in A.

    Args:
        atomic_numbers: The elements to evaluate, one entry each.
        k: Spatial frequency `1/d` in inverse A.

    Returns:
        `sum_i a_i exp(-b_i k^2 / 4)`, with the frequency axes leading. The quarter is the Peng
        tables' own variable, `s = sin(theta)/lambda = k/2`.
    """
    parameters = PengScatteringFactorParameters(np.asarray(atomic_numbers))
    a, b = jnp.asarray(parameters.a), jnp.asarray(parameters.b)
    return (a * jnp.exp(-b * jnp.asarray(k)[..., None, None] ** 2 / 4.0)).sum(-1)


def signal_power(
    atoms: pd.DataFrame, k: Float[Array, " ..."], *, hydrogens_per_heavy_atom: float | None = None
) -> Float[Array, " ..."]:
    """`S(k)`, the particle's signal power, in A^2.

    Missing hydrogens are restored by adding amplitudes, not powers; this matters because the module
    is absolute rather than relative.

    Args:
        atoms: Atom table in the `mmdf` schema, as `structures` returns. Only `atomic_number` is
            read, and every row counts once, so pass the assembly and the symmetry multiplicity comes
            with it.
        k: Spatial frequency in inverse A.
        hydrogens_per_heavy_atom: Hydrogens to restore per non-hydrogen atom, added COHERENTLY --
            each heavy atom scatters as `f_e(Z) + n f_e(1)`. Averaged over the structure rather than
            assigned per atom. `None` uses only the hydrogens the table already contains. See
            docs/rationale/core-modules.md.

    Returns:
        `S(k)`, falling by roughly two orders of magnitude between k = 0 and Nyquist.

    Raises:
        ValueError: If `hydrogens_per_heavy_atom` is negative.
    """
    atomic_number = atoms["atomic_number"].to_numpy()
    if hydrogens_per_heavy_atom is None:
        elements, counts = np.unique(atomic_number, return_counts=True)
        return (scattering_factor(elements, k) ** 2 * counts).sum(-1)
    if hydrogens_per_heavy_atom < 0.0:
        raise ValueError(f"hydrogens_per_heavy_atom must be non-negative, got {hydrogens_per_heavy_atom}")
    # hydrogens already present are dropped rather than kept alongside the restored ones: a table
    # that has them would otherwise count them twice, once in its own row and once on its parent
    heavy = atomic_number[atomic_number > 1]
    elements, counts = np.unique(heavy, return_counts=True)
    effective = scattering_factor(elements, k) + hydrogens_per_heavy_atom * scattering_factor(np.array([1]), k)
    return (effective**2 * counts).sum(-1)


def ssnr(
    k: Float[Array, " ..."],
    *,
    atoms: pd.DataFrame,
    plan: CollectionPlan,
    n_particles: float,
    rmsf: float,
    scale: float,
    orientation_factor: float = 1.0,
    hydrogens_per_heavy_atom: float | None = None,
) -> Float[Array, " ..."]:
    """The reconstruction's spectral signal-to-noise, each physical loss applied exactly once.

    Args:
        atoms: Atom table of the whole particle.
        plan: The collection.
        n_particles: Particles in the reconstruction.
        rmsf: Per-axis intra-state RMS fluctuation in A. The repo's own subject: a target that moves
            is damped by `exp(-2 pi^2 rmsf^2 k^2)` no matter how many particles you collect.
        scale: The global constant `K`, from `calibrate_scale`.
        orientation_factor: Sampling compensation factor, 1 for uniform orientations and below it
            for a preferred one.
        hydrogens_per_heavy_atom: Passed to `signal_power`. It must be the SAME value the `scale`
            was calibrated at -- it moves `S(k)`, and `K` would absorb the difference and then fail
            to transfer.

    Returns:
        SSNR(k), to be read against `FULL_DATASET_SSNR`.
    """
    return (
        scale
        * n_particles
        * orientation_factor
        * plan.dose
        * signal_power(atoms, k, hydrogens_per_heavy_atom=hydrogens_per_heavy_atom)
        * effective_transfer_squared(k, plan=plan, rmsf=rmsf)
    )


def predict_resolution(
    *,
    atoms: pd.DataFrame,
    plan: CollectionPlan,
    n_particles: float,
    rmsf: float,
    scale: float,
    orientation_factor: float = 1.0,
    hydrogens_per_heavy_atom: float | None = None,
    threshold: float = FULL_DATASET_SSNR,
) -> float:
    """The resolution this collection would reach, in A.

    Args:
        atoms: Atom table of the whole particle.
        plan: The collection.
        n_particles: Particles in the reconstruction.
        rmsf: Per-axis intra-state RMS fluctuation in A.
        scale: The global constant `K`.
        orientation_factor: Sampling compensation factor.
        hydrogens_per_heavy_atom: Passed to `signal_power`, and it must match what `calibrate_scale`
            produced this `scale` at.
        threshold: SSNR the reconstruction has to reach.

    Returns:
        `1/k` at the HIGHEST k where SSNR reaches `threshold`, floored at `2 * pixel_size` because
        nothing beyond the detector's Nyquist was searched. Infinite if the SSNR never reaches it,
        which is a target this collection cannot reconstruct at all rather than a large number.
    """
    # 2000 nodes between 200 A and the detector's Nyquist, which resolves the crossing to
    # better than the model's own accuracy at every pixel size worth collecting at
    k = jnp.linspace(1.0 / 200.0, plan.nyquist, 2000)
    reached = (
        ssnr(
            k,
            atoms=atoms,
            plan=plan,
            n_particles=n_particles,
            rmsf=rmsf,
            scale=scale,
            orientation_factor=orientation_factor,
            hydrogens_per_heavy_atom=hydrogens_per_heavy_atom,
        )
        >= threshold
    )
    # the LAST crossing, not the first: the CTF and the envelopes can take the SSNR back above the
    # threshold after it has dipped, and the resolution is where it stays below
    last = int(jnp.max(jnp.where(reached, jnp.arange(k.size), -1)))
    return float("inf") if last < 0 else float(1.0 / k[last])


def calibrate_scale(
    resolution: float,
    *,
    atoms: pd.DataFrame,
    plan: CollectionPlan,
    n_particles: float,
    rmsf: float,
    orientation_factor: float = 1.0,
    hydrogens_per_heavy_atom: float | None = None,
    threshold: float = FULL_DATASET_SSNR,
) -> float:
    """The `K` that reproduces a resolution someone actually achieved.

    Exact rather than solved for: the SSNR is linear in `K`, so pinning the crossing to a stated
    resolution is a division.

    Args:
        resolution: The achieved resolution in A, at `n_particles` under `plan`.
        atoms: Atom table of the whole particle.
        plan: The collection that produced it.
        n_particles: Particles that went into it.
        rmsf: Per-axis intra-state RMS fluctuation in A.
        orientation_factor: Sampling compensation factor.
        hydrogens_per_heavy_atom: Passed to `signal_power`. Record it beside the `K` it produces:
            the two travel together, and `K` calibrated with hydrogens is not `K` without them.
        threshold: SSNR the stated resolution corresponds to.

    Returns:
        `K`. Whether it is the same number on the next target is the model's whole claim.

    Raises:
        ValueError: If the calibrated `K` does not round-trip through `predict_resolution` (which
            returns the HIGHEST crossing while this pins ONE frequency). A single-defocus plan is not
            monotone past the stated resolution, so calibrate against a defocus range.
    """
    common: dict[str, Any] = {
        "atoms": atoms,
        "plan": plan,
        "n_particles": n_particles,
        "rmsf": rmsf,
        "orientation_factor": orientation_factor,
        "hydrogens_per_heavy_atom": hydrogens_per_heavy_atom,
    }
    scale = float(threshold / ssnr(jnp.asarray(1.0 / resolution), scale=1.0, **common))
    # ROUND-TRIPPED: pinning one point and reading back the LAST crossing are not inverse operations
    achieved = predict_resolution(scale=scale, threshold=threshold, **common)
    if abs(achieved - resolution) > 0.01 * resolution:
        raise ValueError(
            f"calibrating at {resolution:.4g} A gives K = {scale:.6g}, which predicts {achieved:.4g} A "
            "instead. Calibrate against a plan with a defocus RANGE, or state a resolution inside "
            "the monotone part of the curve."
        )
    return scale
