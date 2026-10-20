"""The session as planned, before it happens.

The package's configuration root: a plan is read by `particles`, `forward`, `resolution` and every
rung. It imports nothing from the package, so a module at any depth can describe its microscope
without reaching across a subpackage.
"""

from dataclasses import dataclass

DEFAULT_DEFOCUS_RANGE: tuple[float, float] = (5000.0, 25000.0)
"""The uniform defocus sweep every plan assumes unless told otherwise, in A."""


@dataclass(frozen=True)
class CollectionPlan:
    """What a microscope session would be, stated before it happens.

    Everything the instrument contributes travels as one of these, so a CTF and an envelope cannot be
    evaluated at different voltages, or a transfer at a Nyquist the pixel size contradicts.

    Args:
        dose: Total electron exposure in e/A^2. Enters twice and in opposite directions: linearly in
            the SSNR, and through `resolution.radiation_envelope`, which damps high k the longer you
            expose.
        pixel_size: Pixel size in A, which sets `nyquist` and so the ceiling on any resolution.
        defocus_range: Lowest and highest underfocus across the dataset, in A. A range, not a value:
            the spread is what fills the CTF zeros.
        voltage_in_kv: Accelerating voltage.
        spherical_aberration_in_mm: Cs.
        amplitude_contrast_ratio: Q.
    """

    dose: float
    pixel_size: float = 1.0
    defocus_range: tuple[float, float] = DEFAULT_DEFOCUS_RANGE
    voltage_in_kv: float = 300.0
    spherical_aberration_in_mm: float = 2.7
    amplitude_contrast_ratio: float = 0.07

    def __post_init__(self) -> None:
        """Refuse a plan that would silently mislead: a reversed defocus range under-bins the CTF
        spread to a single group (`defocus_groups_for`), and a non-positive pixel or dose breaks the
        Nyquist and SSNR scalings every rung reads off this plan."""
        low, high = self.defocus_range
        if not low <= high:
            raise ValueError(f"defocus_range must be (low, high) with low <= high, got {self.defocus_range}")
        if self.pixel_size <= 0.0:
            raise ValueError(f"pixel_size must be positive, got {self.pixel_size}")
        if self.dose <= 0.0:
            raise ValueError(f"dose must be positive, got {self.dose}")

    @property
    def nyquist(self) -> float:
        """The detector's Nyquist frequency in inverse A, `0.5 / pixel_size`."""
        return 0.5 / self.pixel_size

    @property
    def mid_defocus(self) -> float:
        """The middle of `defocus_range` in A, where the MODEL-ONLY rungs are evaluated.

        Those rungs (`local_information`) take one `CTFParams` rather than a spread, wanting a
        representative operating point; the pool rungs and the fit already see the real spread.
        Shared across call sites so the numbers `targets.attach_marginal` compares cannot drift apart.

        Returns:
            `0.5 * (low + high)` in A. A plan whose range is a single repeated defocus returns it.
        """
        low, high = self.defocus_range
        return 0.5 * (low + high)
