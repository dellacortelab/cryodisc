"""The local rung: conformational information with the pose known only to first order.

`crlb` is the whitened least-squares residual of the difference image `g_z = mu_A - mu_B` on the five
pose directions (two translations, three rotations) -- the appendix's `d^2_CRLB` (A9)-(A11), which
`chain.ChainResult` calls `local_pose`; `known_pose` is the ceiling `I_zz Delta z^2` of (A13).
Whitening against sigma^2(k), transfer inside the model image, DC dropped. Rotations use CENTRAL
differences at 1e-3 rad (usable 1e-4 to 1e-2); do NOT use autodiff, silently wrong at a grid pose.

Rationale: docs/rationale/rungs/local.md
"""

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.rotations import SO3
from jaxtyping import Array, Complex, Float

from cryodisc.forward.cryojax_model import CryoJAXModel
from cryodisc.forward.optics import CTFParams
from cryodisc.grids.fourier import frequency_vectors, whitening_weight
from cryodisc.grids.poses import rotation_of

# the reference pose, then +h and -h about each of the three lab axes
_TANGENTS = jnp.concatenate([jnp.zeros((1, 3)), jnp.eye(3), -jnp.eye(3)])

_POSE_CHUNK = 9
"""Poses rendered at once, which bounds peak memory here.

Each pose renders 7 tangents for two states and the atom render contracts over every atom, so the
separable kernel `(poses * 7, extent, atoms)` can exceed device memory whole. Peak memory is LINEAR
in this number and nothing else here is, so chunking the pose axis is the whole fix. Poses are
independent, so this changes batching not arithmetic: `known_pose` is bit-identical at every chunk
size, `crlb` at this one -- below about 3 the QR picks a different Householder kernel and `crlb`
shifts slightly. `test_local.py` pins both.
"""


class LocalRung(eqx.Module):
    """Per-pose squared distances under two pose assumptions.

    Args:
        known_pose: d^2 with the pose given exactly, the appendix's ceiling `I_zz Delta z^2`.
        crlb: d^2 with the five infinitesimal pose directions removed -- `d^2_CRLB` of (A9)-(A11),
            `I_{z|phi} Delta z^2`.
    """

    known_pose: Float[Array, " n"]
    crlb: Float[Array, " n"]


def local_information(
    model_a: CryoJAXModel,
    model_b: CryoJAXModel,
    angles: Float[Array, "n 3"],
    noise_spectrum: Float[Array, "y x"],
    *,
    contrast: float = 1.0,
    ctf_params: CTFParams | None = None,
    step: float = 1e-3,
    chunk: int = _POSE_CHUNK,
) -> LocalRung:
    """Per-pose `d^2` before and after removing the infinitesimal pose directions.

    Args:
        model_a: Forward model of state A.
        model_b: Forward model of state B, on the same image config.
        angles: (n, 3) reference poses as Euler angles in degrees, cryoJAX intrinsic zyz.
        noise_spectrum: sigma^2(k) on the unshifted fft grid, as `particles.em.fit_noise_model` returns it.
        contrast: Template scale `a`. d^2 scales as its square.
        ctf_params: Optics applied to every render. `None` gives bare projections.
        step: Rotation half-step in radians. Usable from 1e-4 to 1e-2, at about a percent.
        chunk: Poses rendered at once. Bounds peak memory without changing the result;
            see `_POSE_CHUNK`.

    Returns:
        The rung, one entry per reference pose.
    """
    if angles.shape[-1] != 3:
        raise ValueError(f"angles must be (n, 3) Euler triples, got {angles.shape}")
    if angles.shape[0] < 1:
        raise ValueError("the pose bank is empty; there is nothing to average over")
    # a non-finite or non-positive mode makes the whitened norm NaN/inf and every d^2 with it; the
    # sibling rungs reject the same input rather than return a confident-looking wrong answer
    if not bool(jnp.all(jnp.isfinite(noise_spectrum) & (noise_spectrum > 0.0))):
        raise ValueError("noise_spectrum must be finite and strictly positive at every mode")
    return _local_information(
        model_a, model_b, angles, noise_spectrum, contrast=contrast, ctf_params=ctf_params, step=step, chunk=chunk
    )


@eqx.filter_jit
def _local_information(
    model_a: CryoJAXModel,
    model_b: CryoJAXModel,
    angles: Float[Array, "n 3"],
    noise_spectrum: Float[Array, "y x"],
    *,
    contrast: float = 1.0,
    ctf_params: CTFParams | None = None,
    step: float = 1e-3,
    chunk: int = _POSE_CHUNK,
) -> LocalRung:
    """`local_information` once its arguments are checked."""
    # contrast whitening, the square root of the shared weight, so a Euclidean norm is the metric
    weight = jnp.sqrt(whitening_weight(noise_spectrum))
    # the shared signed grid, so the translation tangents below cannot disagree with the whitening
    # weight above about where Nyquist is or which axis is which
    frequency_y, frequency_x = frequency_vectors(noise_spectrum.shape[-1])

    def whiten(image: Complex[Array, "y x"]) -> Float[Array, " m"]:
        """Whiten and stack Re, Im, so the Euclidean norm squared is the whitened inner product."""
        weighted = image * weight
        return jnp.concatenate([weighted.real.ravel(), weighted.imag.ravel()])

    def render_perturbed(model: CryoJAXModel, pose_angles: Float[Array, " 3"]) -> Complex[Array, "7 y x"]:
        """The reference pose and its six rotational neighbors, in Fourier space."""
        base = rotation_of(pose_angles)

        def one(tangent: Float[Array, " 3"]) -> Complex[Array, "y x"]:
            rotation = SO3.exp(step * tangent).compose(base)
            return jnp.fft.fft2(model.render_at_pose(cxs.QuaternionPose(wxyz=rotation.wxyz), ctf_params))

        return eqx.filter_vmap(one)(_TANGENTS)

    def at_pose(pose_angles: Float[Array, " 3"]) -> tuple[Array, Array]:
        images_a = render_perturbed(model_a, pose_angles)
        images_b = render_perturbed(model_b, pose_angles)
        # projection is linear in the potential, so averaging the images averages the volumes
        operating = 0.5 * contrast * (images_a + images_b)
        motion = contrast * (images_a[0] - images_b[0])

        translations = [(-2j * jnp.pi * frequency_x) * operating[0], (-2j * jnp.pi * frequency_y) * operating[0]]
        rotations = (operating[1:4] - operating[4:7]) / (2.0 * step)

        target = whiten(motion)
        # QR because only the SPAN matters, and it is better conditioned than the normal equations
        basis, _ = jnp.linalg.qr(jnp.stack([whiten(g) for g in [*translations, *rotations]], axis=1))
        along_pose = basis.T @ target
        known = target @ target
        return known, known - along_pose @ along_pose

    # `lax.map` with a batch size vmaps within a chunk and scans across chunks, freeing each chunk's
    # renders before the next is built; a plain vmap would materialize all at once. See `_POSE_CHUNK`.
    known_pose, crlb = jax.lax.map(at_pose, angles, batch_size=min(chunk, angles.shape[0]))
    return LocalRung(known_pose=known_pose, crlb=jnp.maximum(crlb, 0.0))
