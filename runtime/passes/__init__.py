from .passes import (
    Pass, PassResult,
    DEFAULT_PIPELINE, PRE_RESOLUTION_PIPELINE, POST_RESOLUTION_PIPELINE,
    run_pipeline, run_until_stable,
    absorb_into_matmul, constant_fold, absorb_mask_into_attention,
    merge_parallel_matmuls, eliminate_dead_code,
)
from .fusion import (
    fuse, fuse_dags, FusionPattern, FUSION_PATTERNS, register_fusion,
)
