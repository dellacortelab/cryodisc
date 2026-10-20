"""Certify one template bank against another: does THIS renderer agree with a reference?

Three statistics, each blind to what the others catch: `correlation` (per matched template,
translation-registered) catches a pose permutation or mispaired bank; `amp_ratio` is the per-shell
LEVEL, translation-invariant; `frc` the per-shell SHAPE, after sub-pixel registration. With amplitude
ratio `A(k)` and correlation `c(k)`, the relative squared error is `(A - c)^2 + (1 - c^2)`, so `A`
can sit at 1 while `c` collapses. Compare NOISE-FREE templates. Rationale: docs/rationale/validation/bank.md
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from cryodisc.grids.fourier import frequency_grid, phase_ramp, shell_indices, shell_metrics, shell_sum
from cryodisc.microscope import aberration_coefficients, contrast, wavelength

if TYPE_CHECKING:
    from reproduction.validation.template_bank import TemplateBank

D2_TOL = 0.05
"""Worst-case per-shell deviation a certified bank may carry, as a fraction of `d^2`.

Bounding `max_k |A_k^2 - 1|` bounds the `d^2` error for every shell weighting, which is why the
criterion is per-shell and worst-case rather than an average. See docs/rationale/validation/bank.md.
"""

_UPSAMPLE = 32
"""Registration grid, `1/_UPSAMPLE` px. Sets the shape criterion's floor in closed form."""

_FINE_WINDOW = 1.5
"""Half-width in pixels of `register_subpixel`'s fine search, about the integer peak.

At or above one covers the whole residual: the coarse stage has taken the integer peak, so what is
left is under half a pixel per axis. A constant, not an argument: below one the search CLAMPS and
returns a wrong alignment silently.
"""

_NCC_GATE = 0.98
"""Median matched correlation a bank must reach. Necessary, and on its own nowhere near sufficient."""


@dataclass(frozen=True)
class BankComparison:
    """One bank measured against another, shell by shell and template by template.

    Args:
        k_cyc_px: Shell center frequencies in cycles per pixel.
        n_modes: Modes in each shell. A shell holding too few is excluded from the verdict.
        amp_ratio: `sqrt(P_a / P_b)` per shell, after dividing by `scale`. The LEVEL.
        frc: `Re<F_a, F_b> / sqrt(P_a P_b)` per shell, after sub-pixel registration. The SHAPE.
        rel_sq_err: `A^2 - 2 c A + 1`, the two combined.
        power_frac: Each shell's share of the reference bank's power. Reported, and deliberately NOT used
            to exclude anything.
        amp_low: 5th percentile of the per-template amplitude ratio. Catches a defect in a few
            templates that `amp_ratio` sums away.
        amp_high: 95th percentile of the same.
        correlation: Registered NCC per matched template.
        scale: The one scalar fitted over the band, `sqrt(sum P_a / sum P_b)`, which makes a
            frequency-FLAT amplitude error invisible here. Read this value in every case.
        shift: `(n, 2)` sub-pixel offsets, as (y, x) in pixels, that aligned each reference template.
        shape_floor: The `1 - c^2` the registration residual alone produces at each shell's outer
            edge. Closed form from `1/(2 _UPSAMPLE)`.
        usable: Shells with enough modes and a reference that is not numerically silent.
        pixel_size: Pixel size in A, when one was given.
        k_max_cyc_px: The sampling limit shells were judged against, when one was applied.
    """

    k_cyc_px: np.ndarray
    n_modes: np.ndarray
    amp_ratio: np.ndarray
    frc: np.ndarray
    rel_sq_err: np.ndarray
    power_frac: np.ndarray
    amp_low: np.ndarray
    amp_high: np.ndarray
    correlation: np.ndarray
    scale: float
    shift: np.ndarray
    shape_floor: np.ndarray
    usable: np.ndarray
    pixel_size: float | None
    k_max_cyc_px: float | None = None

    @property
    def resolution(self) -> np.ndarray:
        """Shell centers as a resolution in A, or NaN throughout when no pixel size was given."""
        if self.pixel_size is None:
            return np.full_like(self.k_cyc_px, np.nan)
        with np.errstate(divide="ignore"):
            return np.where(self.k_cyc_px > 0, self.pixel_size / self.k_cyc_px, np.inf)


@dataclass(frozen=True)
class BankVerdict:
    """Whether a bank may stand in for its reference, and over what band.

    Args:
        passes: Both criteria met in every usable shell, and the median correlation at the gate.
        passes_correlation: Median matched NCC at the gate. Catches a permutation; sees no level error.
        passes_band: Every usable shell within `d2_tol` on both level and shape.
        level_worst: Largest `|A^2 - 1|` over the usable shells.
        shape_worst: Largest `1 - c^2` over the usable shells.
        failing_level: Indices of shells that failed the level criterion.
        failing_shape: Indices that failed the shape criterion.
        certified_from_k: INNER edge of the certified band, in cycles per pixel. Zero when the lowest
            shell was judged. The band is an INTERVAL, not a ceiling.
        certified_k: OUTER edge of the same band. Shells inside it that were not judged are the CTF
            zeros; `usable` says which. The useful output when the gate FAILS.
        certified_resolution: `certified_k` as a resolution in A, or `inf` when nothing is certified.
        certified_low_resolution: `certified_from_k` likewise, or `inf` when the band starts at DC.

    Rationale: docs/rationale/validation/bank.md
    """

    passes: bool
    passes_correlation: bool
    passes_band: bool
    level_worst: float
    shape_worst: float
    failing_level: np.ndarray
    failing_shape: np.ndarray
    certified_from_k: float
    certified_k: float
    certified_resolution: float
    certified_low_resolution: float


@jax.jit
def matched_correlations(bank_a: Float[Array, "n y x"], bank_b: Float[Array, "n y x"]) -> Float[Array, " n"]:
    """Registered normalized cross-correlation of `bank_a[i]` against `bank_b[i]`, in one jitted pass.

    Each image is mean-subtracted and the correlation maximized over integer circular shifts, so a
    translation offset is not counted as an optics failure. The zero-norm guard is a `where` (no
    scalar to branch on under `vmap`), and the denominator is floored before the divide because
    `where` selects after both branches are evaluated.

    Args:
        bank_a: One bank, `(n, box, box)`.
        bank_b: The other, same shape and already registered against it -- INTEGER shifts only here.

    Returns:
        `(n,)` correlations, zero wherever either image is constant.
    """

    def one(image_a: Float[Array, "y x"], image_b: Float[Array, "y x"]) -> Float[Array, ""]:
        a = image_a - jnp.mean(image_a)
        b = image_b - jnp.mean(image_b)
        norm = jnp.linalg.norm(a) * jnp.linalg.norm(b)
        peak = jnp.max(jnp.fft.ifft2(jnp.fft.fft2(a) * jnp.conj(jnp.fft.fft2(b))).real)
        return jnp.where(norm == 0.0, 0.0, peak / jnp.where(norm == 0.0, 1.0, norm))

    return jax.vmap(one)(bank_a, bank_b)


@jax.jit
def register_subpixel(image_a: Float[Array, "y x"], image_b: Float[Array, "y x"]) -> Float[Array, " 2"]:
    """The offset `image_b` must be shifted by to align with `image_a`, to `1/_UPSAMPLE` px.

    Two stages, both required: the integer cross-correlation peak first (the fine search covers only
    `+-_FINE_WINDOW` px and otherwise CLAMPS), then the sub-pixel residual -- integer registration
    alone leaves a half-pixel phase error `cos(2 pi k.t)`. The fine stage evaluates
    `CC(t) = sum_k F_a conj(F_b) exp(2 pi i k.t)` on a grid fixed at `_UPSAMPLE`, so its residual
    bound matches the `shape_floor` `compare_banks` derives from the same constant.

    Args:
        image_a: The reference.
        image_b: The image to align to it.

    Returns:
        `(dy, dx)` in pixels, ready for `fourier.phase_ramp`.
    """
    box_size = image_a.shape[-1]
    cross = jnp.fft.fft2(image_a - jnp.mean(image_a)) * jnp.conj(jnp.fft.fft2(image_b - jnp.mean(image_b)))

    # stage one: the integer peak, wrapped into (-box/2, box/2] so a shift of -3 does not read as +29
    coarse_surface = jnp.fft.ifft2(cross).real
    coarse_flat = jnp.argmax(coarse_surface)
    coarse = jnp.stack([coarse_flat // box_size, coarse_flat % box_size]).astype(jnp.float32)
    coarse = jnp.where(coarse > box_size / 2, coarse - box_size, coarse)

    # stage two: the residual, on a fine grid centered on that peak
    residuals = jnp.arange(-_FINE_WINDOW, _FINE_WINDOW + 0.5 / _UPSAMPLE, 1.0 / _UPSAMPLE)
    frequency = jnp.fft.fftfreq(box_size)
    rows = jnp.exp(2j * jnp.pi * jnp.outer(coarse[0] + residuals, frequency))
    columns = jnp.exp(2j * jnp.pi * jnp.outer(coarse[1] + residuals, frequency))
    surface = jnp.real(rows @ cross @ columns.T)
    flat = jnp.argmax(surface)
    return coarse + jnp.stack([residuals[flat // residuals.size], residuals[flat % residuals.size]])


def shell_transfer_rms(
    box_size: int,
    pixel_size: float,
    defocus_in_angstroms: float,
    voltage_in_kv: float,
    *,
    n_shells: int,
    spherical_aberration_in_mm: float = 2.7,
    amplitude_contrast_ratio: float = 0.07,
) -> np.ndarray:
    """RMS |CTF| in each shell -- how much of what it is given the microscope actually transfers.

    The level statistic is a ratio of two amplitudes, so near a CTF zero it divides two small numbers
    and amplifies any engine difference without bound -- optics, not a bank defect. Computed from the
    OPTICS and never from either bank, which separates it from excluding shells by measured power.

    Args:
        box_size: Templates are `(box, box)`.
        pixel_size: A/px.
        defocus_in_angstroms: Underfocus, positive.
        voltage_in_kv: Accelerating voltage.
        n_shells: Radial shells, matching the comparison's.
        spherical_aberration_in_mm: Cs. Affects only which shells are judged, not what is measured.
        amplitude_contrast_ratio: Q. Sets the transfer as k goes to zero, where phase contrast vanishes.

    Returns:
        `(n_shells,)` RMS |CTF|, in [0, 1].
    """
    index, _ = shell_indices(box_size, n_shells)
    frequency = np.asarray(frequency_grid(box_size, pixel_size))
    a, b = aberration_coefficients(
        jnp.asarray(frequency**2),
        voltage_in_kv=voltage_in_kv,
        spherical_aberration_in_mm=spherical_aberration_in_mm,
    )
    transfer = np.asarray(contrast(a * defocus_in_angstroms + b, amplitude_contrast_ratio))

    # both sliced to `n_shells`, dropping the out-of-band bin the box corners fall in; see
    # `fourier.shell_sum`
    squared = shell_sum(jnp.asarray(transfer**2), index, n_shells)[:n_shells]
    counts = shell_sum(jnp.ones(index.shape, jnp.float32), index, n_shells)[:n_shells]
    # an EMPTY shell reads 0.0, not NaN: compared against `min_transfer`, 0.0 excludes the shell
    # rather than propagating a NaN into the comparison
    return np.asarray(jnp.where(counts > 0, jnp.sqrt(squared / jnp.where(counts > 0, counts, 1.0)), 0.0))


def ctf_sampling_limit(box_size: int, pixel_size: float, defocus_in_angstroms: float, voltage_in_kv: float) -> float:
    """Highest frequency at which this box still SAMPLES the CTF, in cycles per pixel.

    Beyond it the transfer aliases and a ratio measures the sampling, not the physics. From
    `chi = pi lambda df k^2`, adjacent samples `dk = 1/(N p)` differ by less than half a cycle when

        k_max = N p^2 / (2 lambda df)

    Args:
        box_size: Templates are `(box, box)`.
        pixel_size: A/px.
        defocus_in_angstroms: The defocus both banks were rendered at. Larger defocus pushes the
            limit to LOWER frequency.
        voltage_in_kv: Accelerating voltage, for the wavelength.

    Returns:
        The limit in cycles per pixel, capped at Nyquist.
    """
    lam = float(wavelength(voltage_in_kv))
    limit = box_size * pixel_size**2 / (2.0 * lam * abs(defocus_in_angstroms))
    return float(min(limit, 0.5))


def compare_banks(
    bank_a: Float[Array, "n y x"],
    bank_b: Float[Array, "n y x"],
    *,
    n_shells: int = 8,
    pixel_size: float | None = None,
    fit_scale: bool = True,
    min_modes: int = 8,
    k_max_cyc_px: float | None = None,
    transfer_rms: np.ndarray | None = None,
    min_transfer: float = 0.0,
) -> BankComparison:
    """Measure `bank_a` against `bank_b`, template by template and shell by shell.

    Args:
        bank_a: Bank under test, `(n, box, box)`, real and NOISE-FREE.
        bank_b: Reference bank, same shape, rendered at the same poses, box, pixel and defocus.
        n_shells: Radial shells.
        pixel_size: Pixel size in A, for the resolutions in the report.
        fit_scale: Divide the amplitude ratio by one scalar fitted over the band, making the
            comparison blind to a frequency-flat error. Pass `False` when the units are shared.
        min_modes: Fewest modes a shell may hold and still enter the verdict.
        k_max_cyc_px: Highest frequency to judge, in cycles per pixel. `None` judges to Nyquist.
            Shells above it are excluded from the verdict AND the scale fit. From `ctf_sampling_limit`.
        transfer_rms: Per-shell RMS |CTF| from `shell_transfer_rms`, from the optics not either bank.
            `None` judges every shell.
        min_transfer: Shells transferring less than this are excluded: an amplitude ratio near a CTF
            zero swings with the binning rather than the forward model.

    Returns:
        The comparison.

    Raises:
        ValueError: If the banks disagree in shape, are not `(n, box, box)` with a square box, or
            are empty.
    """
    under_test, reference = jnp.asarray(bank_a), jnp.asarray(bank_b)
    if under_test.shape != reference.shape:
        raise ValueError(f"bank shapes differ: {under_test.shape} vs {reference.shape}")
    if under_test.ndim != 3 or under_test.shape[1] != under_test.shape[2]:
        raise ValueError(f"banks must be (n, box, box) with a square box, got {under_test.shape}")
    if under_test.shape[0] == 0:
        raise ValueError("the banks are empty; there is nothing to compare")

    box_size = under_test.shape[-1]
    shift = jax.vmap(register_subpixel)(under_test, reference)
    # only the REFERENCE is centered, so the registration has a mean-free image to align.
    # `shell_metrics` subtracts each image's mean before transforming, so centering here would be a no-op.
    centered = reference - jnp.mean(reference, axis=(-2, -1), keepdims=True)
    aligned = jnp.fft.ifft2(jnp.fft.fft2(centered) * phase_ramp(shift, box_size)).real

    # ONE pass: the per-template ratios behind `amp_low`/`amp_high` come back from the same
    # `shell_metrics` call, being the same per-image shell sums before the reduction over the bank.
    shells = shell_metrics(under_test, aligned, n_shells=n_shells)
    per_template = shells["amp_ratio_per_image"]

    n_modes = np.asarray(shells["n_modes"])
    amp_ratio = np.asarray(shells["amp_ratio"])
    power_frac = np.asarray(shells["power_frac"])
    # a shell with FEW modes is excluded, one with little POWER is not: a real defect can live in a
    # low-power shell
    usable = (n_modes >= min_modes) & np.isfinite(amp_ratio)
    if k_max_cyc_px is not None:
        usable = usable & (np.asarray(shells["k_cyc_px"]) <= k_max_cyc_px)
    if transfer_rms is not None and min_transfer > 0.0:
        usable = usable & (np.asarray(transfer_rms) >= min_transfer)

    scale = 1.0
    if fit_scale and usable.any():
        # power-weighted least squares over the certifiable band, which is what `sqrt(sum/sum)` is
        weight = power_frac[usable]
        ratio_squared = amp_ratio[usable] ** 2
        scale = float(np.sqrt(np.sum(weight * ratio_squared) / max(np.sum(weight), 1e-300)))
        scale = scale if np.isfinite(scale) and scale > 0.0 else 1.0

    edges = np.linspace(0.0, 0.5, n_shells + 1)
    return BankComparison(
        k_cyc_px=np.asarray(shells["k_cyc_px"]),
        n_modes=n_modes,
        amp_ratio=amp_ratio / scale,
        frc=np.asarray(shells["frc"]),
        # RECOMPUTED at the fitted scale, not taken from `shell_metrics`, whose ratio is RAW: with
        # `fit_scale=True` that one contradicts the `amp_ratio` and `frc` reported beside it.
        rel_sq_err=(amp_ratio / scale) ** 2 - 2.0 * np.asarray(shells["frc"]) * (amp_ratio / scale) + 1.0,
        power_frac=power_frac,
        amp_low=np.nanpercentile(np.asarray(per_template), 5, axis=0) / scale,
        amp_high=np.nanpercentile(np.asarray(per_template), 95, axis=0) / scale,
        # against the ALIGNED reference: `matched_correlations` maximizes over INTEGER shifts only, and
        # a half-pixel offset it cannot undo would fail the gate for a pure centering convention
        correlation=np.asarray(matched_correlations(under_test, aligned)),
        scale=scale,
        shift=np.asarray(shift),
        # the residual is at most 1/(2 _UPSAMPLE) per axis, so sqrt(2)/(2 _UPSAMPLE) in magnitude
        shape_floor=1.0 - np.cos(2.0 * np.pi * edges[1:] * np.sqrt(2.0) / (2.0 * _UPSAMPLE)) ** 2,
        usable=usable,
        k_max_cyc_px=None if k_max_cyc_px is None else float(k_max_cyc_px),
        pixel_size=pixel_size,
    )


def bank_verdict(comparison: BankComparison, *, d2_tol: float = D2_TOL) -> BankVerdict:
    """Pass or fail a comparison, and report the band it can certify either way.

    The criterion, for every usable shell:

        |A(k)^2 - 1| <= d2_tol      the level, which bounds the `d^2` error for every downstream weighting
        1 - c(k)^2   <= d2_tol      the shape, without which incoherent power could offset a level loss

    The shape inherits the level's tolerance; its floor is `comparison.shape_floor`. Blind to: a
    frequency-flat amplitude error (absorbed by `scale`), a pose permutation or mispaired bank (the
    correlation's job), a pure translation, a defect in a few templates (`amp_low`/`amp_high` report
    it), WHICH bank is wrong, and anything above 0.5 cyc/px.

    Args:
        comparison: From `compare_banks`.
        d2_tol: Worst-case per-shell deviation allowed. See `D2_TOL`.

    Returns:
        The verdict.

    Raises:
        ValueError: If no shell is usable, so there is nothing to certify either way.
    """
    usable = comparison.usable
    if not usable.any():
        raise ValueError("no shell holds enough modes to certify; use a larger box or fewer shells")

    level = np.abs(comparison.amp_ratio**2 - 1.0)
    shape = 1.0 - comparison.frc**2
    # `~(<=)` rather than `>`, so a NaN counts as a failure instead of passing every comparison
    failing_level = usable & ~(level <= d2_tol)
    failing_shape = usable & ~(shape <= d2_tol)

    edges = np.linspace(0.0, 0.5, len(level) + 1)
    # the band is [from, to]: its inner edge is the first shell actually JUDGED, which is not DC when
    # `min_transfer` excluded the lowest ones. See `BankVerdict.certified_from_k`.
    certified_from, certified = 0.0, 0.0
    started = False
    for i in range(len(level)):
        # only a JUDGED shell moves the edge; an unusable one is skipped, not broken on, since
        # `min_transfer` passes over CTF zeros and low shells where phase contrast vanishes
        if not usable[i]:
            continue
        if failing_level[i] or failing_shape[i]:
            break
        if not started:
            certified_from, started = edges[i], True
        certified = edges[i + 1]
    # clamped to the limit the shell was ADMITTED under: `usable` tests a shell's CENTER, so the
    # outermost one straddles `k_max_cyc_px` and its outer edge claims a band that was never sampled
    if comparison.k_max_cyc_px is not None:
        certified = min(certified, float(comparison.k_max_cyc_px))

    def as_resolution(k: float) -> float:
        """`k` in cycles per pixel as a resolution in A, or `inf` when there is no band to convert."""
        return float(comparison.pixel_size / k) if comparison.pixel_size is not None and k > 0.0 else np.inf

    passes_correlation = bool(np.median(comparison.correlation) >= _NCC_GATE)
    passes_band = not (failing_level.any() or failing_shape.any())
    return BankVerdict(
        passes=passes_correlation and passes_band,
        passes_correlation=passes_correlation,
        passes_band=passes_band,
        level_worst=float(np.nanmax(level[usable])),
        shape_worst=float(np.nanmax(shape[usable])),
        failing_level=np.flatnonzero(failing_level),
        failing_shape=np.flatnonzero(failing_shape),
        certified_from_k=float(certified_from),
        certified_k=float(certified),
        certified_resolution=as_resolution(certified),
        certified_low_resolution=as_resolution(certified_from),
    )


def compare_engines(
    under_test: "TemplateBank",
    reference: "TemplateBank",
    *,
    n_shells: int = 8,
    min_transfer: float = 0.5,
) -> BankComparison:
    """`compare_banks` between two rendered banks, with their provenance checked first.

    Prefer this over the array form: the gate measures two ENGINES only when the box, pixel, voltage,
    defocus and poses are shared, and `assert_comparable` makes that a precondition. A pose mismatch
    reads as a forward-model deficit that no spectral statistic can see.

    Args:
        under_test: The bank being certified, usually the fast one.
        reference: What it is measured against, usually multislice. `pixel_size` for the report
            comes from here.
        n_shells: Radial shells.
        min_transfer: Fewest RMS |CTF| a shell may transfer and still enter the verdict. Defaulted to
            a half so the level is judged only where the microscope passes at least half of what it
            can. Pass 0.0 to judge every shell.

    Returns:
        The comparison, ready for `bank_verdict`.

    Raises:
        ValueError: If the two banks were not rendered comparably, naming which setting disagreed.
    """
    under_test.assert_comparable(reference)
    # derived here, not asked of the caller: the band is a fact about the banks, which carry the
    # defocus, box and pixel
    limit, transfer = None, None
    if reference.defocus_in_angstroms is not None:
        limit = ctf_sampling_limit(
            reference.box_size, reference.pixel_size, reference.defocus_in_angstroms, reference.voltage_in_kv
        )
        transfer = shell_transfer_rms(
            reference.box_size,
            reference.pixel_size,
            reference.defocus_in_angstroms,
            reference.voltage_in_kv,
            n_shells=n_shells,
        )
    return compare_banks(
        under_test.images,
        reference.images,
        n_shells=n_shells,
        pixel_size=reference.pixel_size,
        k_max_cyc_px=limit,
        transfer_rms=transfer,
        min_transfer=min_transfer,
    )
