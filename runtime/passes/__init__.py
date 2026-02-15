from .passes import (
    Pass, PassResult, DEFAULT_PIPELINE, run_pipeline, run_until_stable,
    absorb_into_matmul, constant_fold, absorb_mask_into_attention,
    eliminate_dead_code,
)
from .fusion import fuse, FusionPattern, FUSION_PATTERNS, register_fusion
