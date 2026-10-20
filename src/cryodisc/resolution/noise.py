"""sigma^2(k) for a stated collection, in the units cryoJAX renders in.

cryoJAX's `quantity_mode="contrast"` returns `C` with detector-plane intensity `|psi|^2 = 1 + 2C`, so

    sigma^2(k) = box_size^2 / (4 dose pixel_size^2)  /  [DQE(k) E_rad(k)^2 E_conf(k)^2]

with the `box_size^2` of the unnormalized `fft2` and no leftover constant, so the matching template
scale is `contrast = 1`. Envelopes come from `transfer.envelope_squared`; the CTF is absent because
`CTFParams` put it in the template. Render templates WITH a `CTFParams`, or bare projections overstate the signal.
"""

from jaxtyping import Array, Float

from cryodisc.grids.fourier import frequency_grid
from cryodisc.microscope import CollectionPlan
from cryodisc.resolution.transfer import envelope_squared


def noise_spectrum(plan: CollectionPlan, box_size: int, *, rmsf: float = 0.0) -> Float[Array, "y x"]:
    """The noise a collection buys, per fft mode, against a cryoJAX contrast template.

    Args:
        plan: The collection. The dose and the pixel size set the shot-noise floor; the pixel size
            also sets the detector's Nyquist, which is where DQE is read against.
        box_size: Image size in pixels, which sets the `fft2` normalization and so the level.
        rmsf: Per-axis intra-state RMS fluctuation in A. Zero leaves `E_conf` at one, which is right
            only for a state with no internal disorder.

    Returns:
        sigma^2(k) on the unshifted fft grid, to be paired with `contrast = 1` and with templates
        rendered THROUGH a `CTFParams`, which is what makes them contrast. Largest at high k, where
        DQE and the radiation envelope have both fallen.
    """
    shot = box_size**2 / (4.0 * plan.dose * plan.pixel_size**2)
    return shot / envelope_squared(frequency_grid(box_size, plan.pixel_size), plan=plan, rmsf=rmsf)
