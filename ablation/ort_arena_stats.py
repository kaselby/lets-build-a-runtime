"""Query ORT's internal arena allocator stats via the C API.

The Python bindings don't expose AllocatorGetStats, but the C API does.
We load libonnxruntime via ctypes and navigate the OrtApi vtable directly.

The key insight: OrtApi::CreateAllocator(session, mem_info) returns an allocator
that wraps the session's internal arena. Calling GetStats on it gives us MaxInUse —
the peak activation memory that ORT's planner actually allocated.

Offsets are for ORT v1.24.1 (ORT_API_VERSION=24), computed from the header with offsetof().
"""

import ctypes
import ctypes.util
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# OrtApi vtable offsets (ORT v1.24.1, ORT_API_VERSION=24)
# Computed via offsetof(OrtApi, <field>) / sizeof(void*)
# ---------------------------------------------------------------------------

_OFF = {
    "GetErrorMessage": 2,
    "CreateEnv": 3,
    "CreateSession": 7,
    "Run": 9,
    "CreateSessionOptions": 10,
    "EnableMemPattern": 16,
    "EnableCpuMemArena": 18,
    "SetSessionLogSeverityLevel": 22,
    "SetSessionGraphOptimizationLevel": 23,
    "SetIntraOpNumThreads": 24,
    "SessionGetInputCount": 30,
    "SessionGetOutputCount": 31,
    "SessionGetInputName": 36,
    "SessionGetOutputName": 37,
    "CreateRunOptions": 39,
    "CreateTensorWithDataAsOrtValue": 49,
    "CreateCpuMemoryInfo": 69,
    "AllocatorAlloc": 75,
    "AllocatorFree": 76,
    "GetAllocatorWithDefaultOptions": 78,
    "ReleaseEnv": 92,
    "ReleaseStatus": 93,
    "ReleaseMemoryInfo": 94,
    "ReleaseSession": 95,
    "ReleaseValue": 96,
    "ReleaseRunOptions": 97,
    "ReleaseSessionOptions": 100,
    "CreateAllocator": 131,
    "ReleaseAllocator": 132,
    "GetKeyValue": 297,
    "GetKeyValuePairs": 298,
    "ReleaseKeyValuePairs": 300,
    "AllocatorGetStats": 319,
}

# ORT enums
ORT_LOGGING_LEVEL_WARNING = 3
ORT_DEVICE_ALLOCATOR = 0   # OrtDeviceAllocator
ORT_MEM_TYPE_DEFAULT = 0    # OrtMemTypeDefault
ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT = 1
ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 = 7
ORT_ENABLE_ALL = 99  # GraphOptimizationLevel

_NUMPY_TO_ORT_DTYPE = {
    np.float32: ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    np.int64: ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
}


def _find_ort_lib():
    """Find the ORT shared library from the installed Python package."""
    try:
        import onnxruntime
        pkg = Path(onnxruntime.__file__).parent / "capi"
        for f in pkg.iterdir():
            if f.suffix in (".dylib", ".so") and "onnxruntime" in f.name:
                return str(f)
    except ImportError:
        pass
    return None


class OrtCApi:
    """Thin ctypes wrapper around the ORT C API vtable."""

    def __init__(self):
        lib_path = _find_ort_lib()
        if lib_path is None:
            raise RuntimeError("Cannot find libonnxruntime")

        self._lib = ctypes.CDLL(lib_path)

        # OrtGetApiBase() → OrtApiBase*
        # OrtApiBase = { GetApi: func_ptr, GetVersionString: func_ptr }
        self._lib.OrtGetApiBase.restype = ctypes.POINTER(ctypes.c_void_p * 2)
        api_base = self._lib.OrtGetApiBase()

        # GetApi is the first function pointer in OrtApiBase
        get_api = ctypes.cast(
            api_base.contents[0],
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_uint32),
        )

        # GetApi(ORT_API_VERSION) → OrtApi* (array of function pointers)
        api_ptr = get_api(24)  # ORT_API_VERSION for 1.24.1
        if not api_ptr:
            raise RuntimeError("OrtGetApiBase()->GetApi(24) returned NULL")

        # Store as array of void pointers (415 entries for v1.24.1)
        self._api = ctypes.cast(api_ptr, ctypes.POINTER(ctypes.c_void_p * 415)).contents

    def _fn(self, name, restype, *argtypes):
        """Get a function pointer from the vtable by name."""
        idx = _OFF[name]
        raw = self._api[idx]
        if not raw:
            raise RuntimeError(f"OrtApi.{name} (offset {idx}) is NULL")
        ftype = ctypes.CFUNCTYPE(restype, *argtypes)
        return ctypes.cast(raw, ftype)

    def _check(self, status, context=""):
        """Check ORT status — NULL means success, non-NULL is error."""
        if status:
            # GetErrorMessage(status) → const char*
            get_msg = self._fn("GetErrorMessage", ctypes.c_char_p, ctypes.c_void_p)
            msg = get_msg(status)
            if isinstance(msg, bytes):
                msg = msg.decode()
            # Release the status
            release = self._fn("ReleaseStatus", None, ctypes.c_void_p)
            release(status)
            raise RuntimeError(f"ORT error ({context}): {msg}")


def get_ort_arena_stats(onnx_path: str, input_shape: tuple[int, ...],
                        input_dtype=np.float32) -> dict[str, str]:
    """Run an ONNX model through ORT's C API and return arena allocator stats.

    Returns dict of stat name → value string (e.g. {"MaxInUse": "49152", ...}).
    Returns empty dict if stats are not available.
    """
    api = OrtCApi()
    VP = ctypes.c_void_p

    # --- Create environment ---
    env = VP()
    status = api._fn("CreateEnv", VP, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(VP))(
        ORT_LOGGING_LEVEL_WARNING, b"bench", ctypes.byref(env)
    )
    api._check(status, "CreateEnv")

    # --- Session options ---
    opts = VP()
    status = api._fn("CreateSessionOptions", VP, ctypes.POINTER(VP))(ctypes.byref(opts))
    api._check(status, "CreateSessionOptions")

    # Enable arena + memory pattern
    status = api._fn("EnableCpuMemArena", VP, VP, ctypes.c_int)(opts, 1)
    api._check(status, "EnableCpuMemArena")
    status = api._fn("EnableMemPattern", VP, VP, ctypes.c_int)(opts, 1)
    api._check(status, "EnableMemPattern")
    status = api._fn("SetIntraOpNumThreads", VP, VP, ctypes.c_int)(opts, 1)
    api._check(status, "SetIntraOpNumThreads")
    status = api._fn("SetSessionGraphOptimizationLevel", VP, VP, ctypes.c_int)(opts, ORT_ENABLE_ALL)
    api._check(status, "SetSessionGraphOptimizationLevel")
    status = api._fn("SetSessionLogSeverityLevel", VP, VP, ctypes.c_int)(opts, 3)
    api._check(status, "SetSessionLogSeverityLevel")

    # --- Create session ---
    session = VP()
    model_path = onnx_path.encode("utf-8") if isinstance(onnx_path, str) else onnx_path
    status = api._fn("CreateSession", VP, VP, ctypes.c_char_p, VP, ctypes.POINTER(VP))(
        env, model_path, opts, ctypes.byref(session)
    )
    api._check(status, "CreateSession")

    # --- Get input/output names ---
    default_alloc = VP()
    status = api._fn("GetAllocatorWithDefaultOptions", VP, ctypes.POINTER(VP))(
        ctypes.byref(default_alloc)
    )
    api._check(status, "GetAllocatorWithDefaultOptions")

    input_count = ctypes.c_size_t()
    status = api._fn("SessionGetInputCount", VP, VP, ctypes.POINTER(ctypes.c_size_t))(
        session, ctypes.byref(input_count)
    )
    api._check(status, "SessionGetInputCount")

    output_count = ctypes.c_size_t()
    status = api._fn("SessionGetOutputCount", VP, VP, ctypes.POINTER(ctypes.c_size_t))(
        session, ctypes.byref(output_count)
    )
    api._check(status, "SessionGetOutputCount")

    # Get first input name
    input_name_ptr = ctypes.c_char_p()
    status = api._fn("SessionGetInputName", VP, VP, ctypes.c_size_t, VP, ctypes.POINTER(ctypes.c_char_p))(
        session, 0, default_alloc, ctypes.byref(input_name_ptr)
    )
    api._check(status, "SessionGetInputName")
    input_name = input_name_ptr.value

    # Get first output name
    output_name_ptr = ctypes.c_char_p()
    status = api._fn("SessionGetOutputName", VP, VP, ctypes.c_size_t, VP, ctypes.POINTER(ctypes.c_char_p))(
        session, 0, default_alloc, ctypes.byref(output_name_ptr)
    )
    api._check(status, "SessionGetOutputName")
    output_name = output_name_ptr.value

    # --- Create input tensor ---
    mem_info = VP()
    status = api._fn("CreateCpuMemoryInfo", VP, ctypes.c_int, ctypes.c_int, ctypes.POINTER(VP))(
        ORT_DEVICE_ALLOCATOR, ORT_MEM_TYPE_DEFAULT, ctypes.byref(mem_info)
    )
    api._check(status, "CreateCpuMemoryInfo")

    ort_element_type = _NUMPY_TO_ORT_DTYPE.get(np.dtype(input_dtype).type)
    if ort_element_type is None:
        raise ValueError(f"Unsupported input dtype: {input_dtype}")

    if np.issubdtype(input_dtype, np.integer):
        input_data = np.random.randint(0, 1000, size=input_shape).astype(input_dtype)
    else:
        input_data = np.random.randn(*input_shape).astype(input_dtype)
    shape_arr = (ctypes.c_int64 * len(input_shape))(*input_shape)

    input_value = VP()
    status = api._fn(
        "CreateTensorWithDataAsOrtValue", VP,
        VP,              # mem_info
        ctypes.c_void_p, # data
        ctypes.c_size_t, # data_len
        ctypes.POINTER(ctypes.c_int64),  # shape
        ctypes.c_size_t, # shape_len
        ctypes.c_int,    # element_type
        ctypes.POINTER(VP),  # out
    )(
        mem_info,
        input_data.ctypes.data,
        input_data.nbytes,
        shape_arr,
        len(input_shape),
        ort_element_type,
        ctypes.byref(input_value),
    )
    api._check(status, "CreateTensorWithDataAsOrtValue")

    # --- Run inference ---
    run_opts = VP()
    status = api._fn("CreateRunOptions", VP, ctypes.POINTER(VP))(ctypes.byref(run_opts))
    api._check(status, "CreateRunOptions")

    input_names_arr = (ctypes.c_char_p * 1)(input_name)
    output_names_arr = (ctypes.c_char_p * 1)(output_name)
    input_values_arr = (VP * 1)(input_value)
    output_values_arr = (VP * 1)(VP())

    status = api._fn(
        "Run", VP,
        VP,  # session
        VP,  # run_options
        ctypes.POINTER(ctypes.c_char_p),  # input_names
        ctypes.POINTER(VP),               # input_values
        ctypes.c_size_t,                  # input_count
        ctypes.POINTER(ctypes.c_char_p),  # output_names
        ctypes.c_size_t,                  # output_count
        ctypes.POINTER(VP),               # output_values
    )(
        session, run_opts,
        input_names_arr, input_values_arr, 1,
        output_names_arr, 1, output_values_arr,
    )
    api._check(status, "Run")

    # --- Get arena allocator stats ---
    # CreateAllocator wraps the session's internal arena allocator
    arena_alloc = VP()
    status = api._fn("CreateAllocator", VP, VP, VP, ctypes.POINTER(VP))(
        session, mem_info, ctypes.byref(arena_alloc)
    )
    api._check(status, "CreateAllocator")

    kvp = VP()
    status = api._fn("AllocatorGetStats", VP, VP, ctypes.POINTER(VP))(
        arena_alloc, ctypes.byref(kvp)
    )
    api._check(status, "AllocatorGetStats")

    # --- Parse key-value pairs ---
    # GetKeyValue returns const char* (not OrtStatus*), returns NULL if key absent.
    # GetKeyValuePairs returns void. ReleaseKeyValuePairs returns void.
    stats = {}
    if kvp.value:
        # Use GetKeyValue(kvp, key) → const char* for each known stat
        get_val = api._fn("GetKeyValue", ctypes.c_char_p, VP, ctypes.c_char_p)
        for key in (b"Limit", b"InUse", b"MaxInUse", b"TotalAllocated",
                    b"NumAllocs", b"NumReserves", b"NumArenaExtensions",
                    b"NumArenaShrinkages", b"MaxAllocSize"):
            val = get_val(kvp, key)
            if val:
                stats[key.decode()] = val.decode()

        # Release KVP (void return)
        api._fn("ReleaseKeyValuePairs", None, VP)(kvp)

    # --- Cleanup ---
    api._fn("ReleaseAllocator", None, VP)(arena_alloc)
    api._fn("ReleaseValue", None, VP)(output_values_arr[0])
    api._fn("ReleaseValue", None, VP)(input_value)
    api._fn("ReleaseRunOptions", None, VP)(run_opts)
    api._fn("ReleaseMemoryInfo", None, VP)(mem_info)
    api._fn("ReleaseSession", None, VP)(session)
    api._fn("ReleaseSessionOptions", None, VP)(opts)
    api._fn("ReleaseEnv", None, VP)(env)

    return stats


def get_ort_peak_memory(onnx_path: str, input_shape: tuple[int, ...]) -> int | None:
    """Get ORT's peak activation memory (MaxInUse) in bytes. Returns None on failure."""
    try:
        stats = get_ort_arena_stats(onnx_path, input_shape)
        if "MaxInUse" in stats:
            return int(stats["MaxInUse"])
        return None
    except Exception as e:
        print(f"  ORT C API stats failed: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# CLI for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import warnings
    import logging
    import torch
    import torch.nn as nn

    class NaiveTransformerBlock(nn.Module):
        def __init__(self, d_model=64, n_heads=4):
            super().__init__()
            self.d_model = d_model; self.n_heads = n_heads; self.d_k = d_model // n_heads
            self.ln1 = nn.LayerNorm(d_model); self.wq = nn.Linear(d_model, d_model)
            self.wk = nn.Linear(d_model, d_model); self.wv = nn.Linear(d_model, d_model)
            self.wo = nn.Linear(d_model, d_model); self.ln2 = nn.LayerNorm(d_model)
            self.ffn1 = nn.Linear(d_model, d_model * 4)
            self.ffn2 = nn.Linear(d_model * 4, d_model)
        def forward(self, x):
            import math; import torch.nn.functional as F
            B, S, D = x.shape; h = self.ln1(x)
            q = self.wq(h).reshape(B,S,self.n_heads,self.d_k).permute(0,2,1,3)
            k = self.wk(h).reshape(B,S,self.n_heads,self.d_k).permute(0,2,1,3)
            v = self.wv(h).reshape(B,S,self.n_heads,self.d_k).permute(0,2,1,3)
            s = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
            w = F.softmax(s, dim=-1); a = torch.matmul(w, v).permute(0,2,1,3).reshape(B, S, D)
            x = x + self.wo(a)
            return x + self.ffn2(torch.relu(self.ffn1(self.ln2(x))))

    configs = [
        ("Toy",    64,  4,  32),
        ("Small",  256, 4,  128),
        ("Medium", 512, 8,  256),
    ]

    for name, d, h, s in configs:
        model = NaiveTransformerBlock(d, h); model.eval()
        x = torch.randn(1, s, d)

        fd, path = tempfile.mkstemp(suffix=".onnx"); os.close(fd)
        logging.getLogger("torch.onnx").setLevel(logging.ERROR)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o1, o2 = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = open(os.devnull, "w")
            torch.onnx.export(model, (x,), path, input_names=["input"],
                              output_names=["output"], opset_version=18)
            sys.stdout, sys.stderr = o1, o2

        stats = get_ort_arena_stats(path, (1, s, d))
        os.unlink(path)

        print(f"\n{name} (d={d}, h={h}, s={s}):")
        if stats:
            for k, v in sorted(stats.items()):
                print(f"  {k}: {v}")
        else:
            print("  (no stats available)")
