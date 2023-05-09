from yacs.config import CfgNode


_C = CfgNode()

_C.LR = 0.0001
_C.NUM_ITERS = 81
_C.JOINTS2D_VISIB_THRESHOLD = 0.75

_C.LOSS_WEIGHTS = CfgNode()
_C.LOSS_WEIGHTS.JOINTS2D = 1.
_C.LOSS_WEIGHTS.POSE_PRIOR = 0.3
_C.LOSS_WEIGHTS.SHAPE_PRIOR = 1.0


def get_optimise_cfg_defaults():
    return _C.clone()
