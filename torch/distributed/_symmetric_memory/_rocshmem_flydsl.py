"""rocSHMEM device API for FlyDSL kernels. ROCm only.

Call these inside ``@flyc.kernel``. The first call links
``librocshmem_device_<arch>.bc`` and registers ``rocshmem_hipmodule_init``
as the post-load hook, which is the same device bring-up Triton does in
``_rocshmem_triton.py``.

FlyDSL ``ffi`` has no pointer type. Pointer arguments are lowered to i64
and a small wrapper bitcode ``inttoptr``s them before the rocSHMEM symbol.
``$ROCM_PATH/llvm/bin/ld.lld`` must exist; this module points ``ROCM_PATH``
at the prefix that shipped the device bitcode when that linker is there.

``put``, ``get``, ``get_nbi``, ``putmem_signal``, ``barrier_all``, and
``sync_all`` call rocSHMEM workgroup symbols. Every thread in the block
must call them with the same arguments.
"""

import os
import shutil
import tempfile
from typing import Any

import torch


if torch.version.hip is None:
    raise RuntimeError(
        "torch.distributed._symmetric_memory._rocshmem_flydsl is a ROCm-only "
        "module (torch.version.hip is None)."
    )

from torch.distributed._symmetric_memory._rocshmem_triton import RocshmemLibFinder


_WRAPPER_BC: str | None = None

# AMDGPU datalayout of librocshmem_device_*.bc. The wrapper is linked next
# to that bitcode, so it has to use the same layout.
_WRAPPER_LL = r"""
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

declare void @rocshmem_putmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_wg(ptr, ptr, i64, i32)
declare void @rocshmem_getmem_nbi_wg(ptr, ptr, i64, i32)
declare void @rocshmem_putmem_signal_wg(ptr, ptr, i64, ptr, i64, i32, i32)

define void @fly_rocshmem_putmem_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_putmem_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_getmem_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_getmem_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_getmem_nbi_wg(i64 %dest, i64 %source, i64 %nbytes, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  call void @rocshmem_getmem_nbi_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, i32 %pe)
  ret void
}

define void @fly_rocshmem_putmem_signal_wg(i64 %dest, i64 %source, i64 %nbytes, i64 %signal, i64 %sig_val, i32 %sig_op, i32 %pe) {
  %dest_ptr = inttoptr i64 %dest to ptr
  %source_ptr = inttoptr i64 %source to ptr
  %signal_ptr = inttoptr i64 %signal to ptr
  call void @rocshmem_putmem_signal_wg(ptr %dest_ptr, ptr %source_ptr, i64 %nbytes, ptr %signal_ptr, i64 %sig_val, i32 %sig_op, i32 %pe)
  ret void
}
"""


def rocshmem_flydsl_module_init(module_handle: int) -> None:
    """Copy ROCSHMEM_CTX_DEFAULT into a FlyDSL-loaded HIP module."""
    from torch._C._distributed_c10d import _nvshmemx_cumodule_init

    _nvshmemx_cumodule_init(module_handle)


def _device_bc() -> str:
    if not os.environ.get("ROCSHMEM_LIB_DIR") and not os.environ.get("ROCM_PATH"):
        hipcc = shutil.which("hipcc")
        if hipcc:
            prefix = os.path.dirname(os.path.dirname(os.path.realpath(hipcc)))
            os.environ["ROCM_PATH"] = prefix
    return RocshmemLibFinder.find_device_library()


def _ensure_rocm_path(device_bc: str) -> str:
    """Point ROCM_PATH at the prefix whose ld.lld matches the device bitcode."""
    prefix = os.path.dirname(os.path.dirname(os.path.abspath(device_bc)))
    lld = os.path.join(prefix, "llvm", "bin", "ld.lld")
    if os.path.isfile(lld):
        os.environ["ROCM_PATH"] = prefix
        return prefix
    current = os.environ.get("ROCM_PATH") or os.environ.get("ROCM_HOME") or "/opt/rocm"
    if os.path.isfile(os.path.join(current, "llvm", "bin", "ld.lld")):
        os.environ["ROCM_PATH"] = current
        return current
    raise RuntimeError(
        "FlyDSL's gpu-module-to-binary pass runs $ROCM_PATH/llvm/bin/ld.lld. "
        f"That linker was not found next to {device_bc} or under ROCM_PATH={current!r}."
    )


def _llvm_as(prefix: str) -> str:
    for relative in (
        os.path.join("lib", "llvm", "bin", "llvm-as"),
        os.path.join("llvm", "bin", "llvm-as"),
    ):
        path = os.path.join(prefix, relative)
        if os.path.isfile(path):
            return path
    raise RuntimeError(f"llvm-as not found under {prefix}")


def _wrapper_bc(prefix: str) -> str:
    global _WRAPPER_BC
    if _WRAPPER_BC is not None and os.path.isfile(_WRAPPER_BC):
        return _WRAPPER_BC
    cache_dir = os.path.join(tempfile.gettempdir(), "rocshmem_flydsl")
    os.makedirs(cache_dir, exist_ok=True)
    # Each rank compiles in its own process. A shared .bc lets two llvm-as
    # writers truncate each other.
    tag = os.getpid()
    ll_path = os.path.join(cache_dir, f"fly_rocshmem_wrappers.{tag}.ll")
    bc_path = os.path.join(cache_dir, f"fly_rocshmem_wrappers.{tag}.bc")
    with open(ll_path, "w") as ll_file:
        ll_file.write(_WRAPPER_LL)
    import subprocess

    subprocess.check_call([_llvm_as(prefix), ll_path, "-o", bc_path])
    _WRAPPER_BC = bc_path
    return bc_path


def _bind() -> None:
    """Link the device bitcode and the pointer wrapper into the kernel being traced."""
    from flydsl.compiler.kernel_function import CompilationContext

    ctx = CompilationContext.get_current()
    if ctx is None:
        raise RuntimeError("rocSHMEM FlyDSL ops must be called inside @flyc.kernel")
    device_bc = _device_bc()
    prefix = _ensure_rocm_path(device_bc)
    # The wrapper has to be linked first. FlyDSL links bitcode with
    # LinkOnlyNeeded, so rocshmem_* symbols referenced only by the wrapper
    # are dropped if the device bitcode is linked before the wrapper.
    ctx.add_link_lib(_wrapper_bc(prefix))
    ctx.add_link_lib(device_bc)
    if rocshmem_flydsl_module_init not in ctx.post_load_processors:
        ctx.post_load_processors.append(rocshmem_flydsl_module_init)


def _ffi(symbol: str, arg_types: list[str], ret_type: str) -> Any:
    from flydsl.expr.extern import ffi

    return ffi(symbol, arg_types, ret_type)


def _as_addr(value: Any) -> Any:
    """Device pointer of a kernel memref argument, as i64.

    FlyDSL ``ffi`` has no pointer type, and kernel tensors arrive as
    ``!fly.memref`` rather than ``!fly.ptr``.
    """
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import fly as fly_d, llvm as llvm_d
    from flydsl.expr.utils.arith import ArithValue

    raw = value.__extract_to_ir_values__()[0]
    ptr = fly_d.extract_aligned_pointer_as_index(ir.Type.parse("!llvm.ptr<1>"), raw)
    return ArithValue(llvm_d.ptrtoint(ir.IntegerType.get_signless(64), ptr))


def _elem_type(value: Any) -> Any:
    elem = getattr(value, "element_type", None)
    if elem is None:
        return value.dtype
    return elem


def _mem_op(symbol: str, dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    dest_ty = _elem_type(dest)
    source_ty = _elem_type(source)
    if dest_ty != source_ty:
        raise RuntimeError(
            f"dest and source element types must match, got {dest_ty} and {source_ty}"
        )
    _bind()
    _ffi(symbol, ["int64", "int64", "int64", "int32"], "void")(
        _as_addr(dest),
        _as_addr(source),
        nelems * (dest_ty.width // 8),
        pe,
    )


def put(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Put ``nelems`` elements from local ``source`` to ``dest`` on ``pe``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_putmem_wg", dest, source, nelems, pe)


def get(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Get ``nelems`` elements from ``source`` on ``pe`` into local ``dest``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_getmem_wg", dest, source, nelems, pe)


def get_nbi(dest: Any, source: Any, nelems: Any, pe: Any) -> None:
    """Non-blocking get. Call ``quiet`` before reading ``dest``.

    ``dest`` and ``source`` must have the same element type.
    Every thread in the block must call this with the same arguments.
    """
    _mem_op("fly_rocshmem_getmem_nbi_wg", dest, source, nelems, pe)


def putmem_signal(
    dest: Any,
    source: Any,
    nbytes: Any,
    signal: Any,
    sig_val: Any,
    sig_op: Any,
    pe: Any,
) -> None:
    """Put ``nbytes`` bytes and update a remote uint64 signal.

    Every thread in the block must call this with the same arguments.
    """
    _bind()
    _ffi(
        "fly_rocshmem_putmem_signal_wg",
        ["int64", "int64", "int64", "int64", "int64", "int32", "int32"],
        "void",
    )(
        _as_addr(dest),
        _as_addr(source),
        nbytes,
        _as_addr(signal),
        sig_val,
        sig_op,
        pe,
    )


def _void(symbol: str) -> None:
    _bind()
    _ffi(symbol, [], "void")()


def quiet() -> None:
    """Wait for outstanding remote-memory operations."""
    _void("rocshmem_quiet")


def fence() -> None:
    """Order remote-memory operations to each target PE."""
    _void("rocshmem_fence")


def barrier_all() -> None:
    """Workgroup barrier across all PEs.

    Every thread in the block must call this with the same arguments.
    """
    _void("rocshmem_barrier_all_wg")


def sync_all() -> None:
    """Workgroup sync across all PEs.

    Every thread in the block must call this with the same arguments.
    """
    _void("rocshmem_sync_all_wg")


def _i32(symbol: str) -> Any:
    _bind()
    from flydsl.expr.utils.arith import ArithValue

    return ArithValue(_ffi(symbol, [], "int32")())


def my_pe() -> Any:
    """PE number of the caller."""
    return _i32("rocshmem_my_pe")


def n_pes() -> Any:
    """Number of PEs."""
    return _i32("rocshmem_n_pes")


def signal_op(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem has no device-bitcode equivalent for signal_op. "
        "Use rocshmem_uint64_atomic_set or rocshmem_uint64_atomic_add instead."
    )


def alltoall(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem_alltoallmem_wg is not available in current device bitcode. "
        "Use host-side rocshmem_alltoallmem_on_stream instead."
    )


def broadcast(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem_broadcastmem_wg is not available in current device bitcode. "
        "Use host-side rocshmem_broadcastmem_on_stream instead."
    )


def reduce(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(
        "rocshmem team reduce is not available in current device bitcode. "
        "Use host-side rocshmem reduce API instead."
    )


__all__ = [
    "alltoall",
    "barrier_all",
    "broadcast",
    "fence",
    "get",
    "get_nbi",
    "my_pe",
    "n_pes",
    "put",
    "putmem_signal",
    "quiet",
    "reduce",
    "rocshmem_flydsl_module_init",
    "signal_op",
    "sync_all",
]
