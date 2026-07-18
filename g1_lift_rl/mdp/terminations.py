# g1_lift_rl/mdp/terminations.py
"""Termination terms per MDP_SPEC.md section 4. time_out uses Isaac Lab's base
mdp.time_out directly (referenced in TerminationsCfg) -- nothing to wrap here."""
from __future__ import annotations

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv

from ..constants import TABLE_TOP_Z

# Margin matches the precedent in g1_redblock_ext's terminations.py (same value,
# not re-derived here -- it's a termination threshold, not a constants.py fact).
_DROP_MARGIN = 0.10
# Cube launched upward from a violent contact event (measured directly: up to
# +0.7m above the table) -- object_dropped only catches falling BELOW the table,
# never being launched ABOVE it. Without this, an episode with a launched cube
# never terminates, so training keeps observing/rewarding a wildly abnormal
# physical state for many steps -- a very plausible route to the value-loss
# divergence seen in training (9e26 -> inf over a handful of iterations). Margin
# is well above any sane task height (LIFT_CAP=0.12, INSPECT_POS ~0.12 above
# table) so it never clips legitimate lift/inspect behavior.
_LAUNCH_MARGIN = 0.30


def object_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube has fallen well below the table top (dropped or knocked off)."""
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w[:, 2] < (TABLE_TOP_Z - _DROP_MARGIN)


def object_launched(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube rocketed far above any sane task height (contact event, not a real
    lift) -- closes the gap object_dropped leaves for upward launches."""
    obj: RigidObject = env.scene["object"]
    return obj.data.root_pos_w[:, 2] > (TABLE_TOP_Z + _LAUNCH_MARGIN)
