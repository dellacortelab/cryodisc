"""The ladder of discriminabilities, ordered by how much about the pose is assumed known.

Each is a conformational distance (noise variance one, whitened against sigma^2(k), DC dropped), in `chain.ChainResult` order:
    known_pose    the true pose is given                      local.local_information
    local_pose    pose known to first order (d^2_CRLB)        local.local_information
    refined_pose  the pose a matched-filter search recovers   refine.refine_poses
    marginal      the pose integrated against its prior       marginal.pose_log_ratio
    best_match    the pose point-estimated on a noisy image   marginal.pose_log_ratio
    achieved      what a pipeline's posteriors actually carry achieved.achieved_information
"""

from cryodisc.rungs.achieved import (
    Calibration,
    achieved_information,
    calibration_curve,
    fraction_bias,
    pipeline_efficiency,
)
from cryodisc.rungs.local import LocalRung, local_information
from cryodisc.rungs.marginal import (
    LogRatio,
    known_pose_log_ratio,
    pose_log_ratio,
    streamed_pose_log_ratio,
)
from cryodisc.rungs.refine import PoseFit, refine_poses, view_angles

__all__ = [
    "Calibration",
    "LocalRung",
    "LogRatio",
    "PoseFit",
    "achieved_information",
    "calibration_curve",
    "fraction_bias",
    "known_pose_log_ratio",
    "local_information",
    "pipeline_efficiency",
    "pose_log_ratio",
    "refine_poses",
    "streamed_pose_log_ratio",
    "view_angles",
]
