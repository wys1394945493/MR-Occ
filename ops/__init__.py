from .msmv_sampling import MSMV_CUDA, msmv_sampling, msmv_sampling_pytorch
try:
    from .superquadric_splatting.tile_local_aggregate_prob_sq import LocalAggregator
except ImportError as import_error:
    _local_aggregator_import_error = import_error

    class LocalAggregator:  # pragma: no cover - used only before extension build
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "The superquadric splatting CUDA extension is not installed. "
                "Run tools/build_ops.sh first."
            ) from _local_aggregator_import_error

__all__ = [
    "LocalAggregator",
    "MSMV_CUDA",
    "msmv_sampling",
    "msmv_sampling_pytorch",
]
