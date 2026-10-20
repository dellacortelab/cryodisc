"""Conformational discriminability and resolution modeling for single-particle cryo-EM.

Given two conformations, how many particles does telling them apart take? The answer is a squared
distance `d^2` per pose, depressed because part of any conformational difference looks like a small
pose change and is unrecoverable at any particle count.

The package is layered (`grids`, `statistics`, `microscope`, `specimen`, `forward`, `resolution`,
`particles`, `rungs`, `pools`, `pipeline`) with imports running only UP the list. Only the entry
points are re-exported here; everything else is reached by module path.
"""

from cryodisc.microscope import CollectionPlan
from cryodisc.particles.sim import ParticleStack, simulate_stack
from cryodisc.pipeline.chain import Prediction, Rung, predict
from cryodisc.pipeline.fraction import FractionEstimate, estimate_fraction
from cryodisc.pipeline.operating_point import OperatingPoint, rung_ratio
from cryodisc.pools import Pool
from cryodisc.specimen.structures import ConformerPair, align_conformers
from cryodisc.statistics.power import fraction_error, information_per_particle, n_for_error, saturation

__all__ = [
    "CollectionPlan",
    "ConformerPair",
    "FractionEstimate",
    "OperatingPoint",
    "ParticleStack",
    "Pool",
    "Prediction",
    "Rung",
    "align_conformers",
    "estimate_fraction",
    "fraction_error",
    "information_per_particle",
    "n_for_error",
    "predict",
    "rung_ratio",
    "saturation",
    "simulate_stack",
]
