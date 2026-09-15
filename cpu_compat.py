"""
cpu_compat.py — make a CUDA-assuming codebase run on a CPU-only PyTorch build,
without editing the vendored repository.

SCOPE, AND WHY IT IS NARROW
---------------------------
This redirects device placement. It does not change any numerics. Every tensor
that would have been created on a GPU is created on the CPU instead, with the
same dtype, shape and contents.

It is a process-wide patch. `torch.Tensor.to` is rebound for every module in
the interpreter, not just the vendored model. That is acceptable here only
because the clinical geometry path is NumPy/SciPy and never touches torch —
if that ever changes, this needs to be confined to a subprocess instead.

WHAT THE PREVIOUS VERSION MISSED
--------------------------------
1. `torch.cuda.IntTensor(...)` and the nine other legacy typed constructors.
   These are module ATTRIBUTES, not methods, so intercepting `.cuda()` and
   `.to('cuda')` cannot see them. On a CPU build the attribute exists and
   raises only when called, which is why it surfaced deep in a forward pass
   rather than at import.

2. Idempotence. The old patch captured `original_load = torch.load` at call
   time. Running it twice — which `load(force=True)` does — captures the
   ALREADY-PATCHED function and wraps it again. Each re-entry adds a layer,
   and a stack of wrappers rewrites map_location repeatedly. Harmless until
   the day one of the wrappers has a side effect. This version installs once.

3. Factory `device=` keywords. `torch.zeros(n, device='cuda')` never calls
   `.to()` or `.cuda()`, so nothing intercepted it.

4. `torch.load` clobbered a caller's explicit `map_location`. Now it only
   supplies one when the caller did not.

5. cuda utility calls — `empty_cache`, `synchronize`, `set_device` — which
   raise on a CPU build. Now no-ops.

`torch.cuda.is_available()` is deliberately left returning False. Faking it
True makes code take GPU branches that then fail somewhere less obvious.
"""

import torch

_INSTALLED = False
_ORIGINALS = {}

_CUDA_NOOPS = ("empty_cache", "synchronize", "set_device", "init",
               "reset_peak_memory_stats", "reset_max_memory_allocated",
               "manual_seed", "manual_seed_all", "ipc_collect")

_FACTORIES = ("tensor", "zeros", "ones", "empty", "full", "arange",
              "linspace", "logspace", "eye", "rand", "randn", "randint",
              "randperm", "as_tensor", "zeros_like", "ones_like",
              "empty_like", "full_like", "rand_like", "randn_like")


def _to_cpu_device(value):
    """Map anything naming a CUDA device onto CPU; pass everything else through."""
    if isinstance(value, str) and "cuda" in value:
        return "cpu"
    if isinstance(value, torch.device) and value.type == "cuda":
        return torch.device("cpu")
    if isinstance(value, int) and not isinstance(value, bool):
        # a bare int in a device position means "cuda:N"
        return torch.device("cpu")
    return value


def _strip_cuda_kwargs(kwargs):
    if "device" in kwargs and kwargs["device"] is not None:
        kwargs["device"] = _to_cpu_device(kwargs["device"])
    return kwargs


def install():
    """
    Apply the patch. Safe to call repeatedly — only the first call takes
    effect, so a FastAPI reload or `load(force=True)` will not stack wrappers.
    """
    global _INSTALLED
    if _INSTALLED:
        return {"already_installed": True}
    applied = []

    # --- 1. legacy typed tensor constructors -------------------------------
    # Discovered by sweeping the module rather than hardcoding a list, so a
    # constructor this file does not know about is still covered.
    for name in dir(torch.cuda):
        if not name.endswith("Tensor"):
            continue
        cpu_equivalent = getattr(torch, name, None)
        if cpu_equivalent is None:
            continue
        _ORIGINALS[("cuda", name)] = getattr(torch.cuda, name)
        setattr(torch.cuda, name, cpu_equivalent)
        applied.append(f"torch.cuda.{name}")

    # --- 2. cuda utility calls that raise on a CPU build --------------------
    for name in _CUDA_NOOPS:
        if hasattr(torch.cuda, name):
            _ORIGINALS[("cuda", name)] = getattr(torch.cuda, name)
            setattr(torch.cuda, name, lambda *a, **k: None)
            applied.append(f"torch.cuda.{name} -> no-op")
    if hasattr(torch.cuda, "current_device"):
        _ORIGINALS[("cuda", "current_device")] = torch.cuda.current_device
        torch.cuda.current_device = lambda: -1
    if hasattr(torch.cuda, "device_count"):
        _ORIGINALS[("cuda", "device_count")] = torch.cuda.device_count
        torch.cuda.device_count = lambda: 0

    # --- 3. .cuda() -> identity --------------------------------------------
    _ORIGINALS[("Tensor", "cuda")] = torch.Tensor.cuda
    _ORIGINALS[("Module", "cuda")] = torch.nn.Module.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self
    applied.append(".cuda() -> identity")

    # --- 4. .to('cuda') -> .to('cpu') ---------------------------------------
    _orig_tensor_to = torch.Tensor.to
    _ORIGINALS[("Tensor", "to")] = _orig_tensor_to

    def _tensor_to(self, *args, **kwargs):
        args = list(args)
        if args:
            mapped = _to_cpu_device(args[0])
            # only rewrite when the first argument really was a device;
            # .to(dtype) and .to(other_tensor) must pass through untouched
            if isinstance(args[0], (str, torch.device)):
                args[0] = mapped
        return _orig_tensor_to(self, *args, **_strip_cuda_kwargs(kwargs))

    torch.Tensor.to = _tensor_to

    _orig_module_to = torch.nn.Module.to
    _ORIGINALS[("Module", "to")] = _orig_module_to

    def _module_to(self, *args, **kwargs):
        args = list(args)
        if args and isinstance(args[0], (str, torch.device)):
            args[0] = _to_cpu_device(args[0])
        return _orig_module_to(self, *args, **_strip_cuda_kwargs(kwargs))

    torch.nn.Module.to = _module_to
    applied.append(".to('cuda') -> .to('cpu')")

    # --- 5. factory functions with device= ----------------------------------
    for name in _FACTORIES:
        fn = getattr(torch, name, None)
        if fn is None:
            continue
        _ORIGINALS[("torch", name)] = fn

        def _wrap(original):
            def wrapped(*args, **kwargs):
                return original(*args, **_strip_cuda_kwargs(kwargs))
            wrapped.__name__ = getattr(original, "__name__", "wrapped")
            wrapped.__doc__ = getattr(original, "__doc__", None)
            return wrapped

        setattr(torch, name, _wrap(fn))
    applied.append(f"{len(_FACTORIES)} factories: device='cuda' -> 'cpu'")

    # --- 6. torch.load ------------------------------------------------------
    _orig_load = torch.load
    _ORIGINALS[("torch", "load")] = _orig_load

    def _load(*args, **kwargs):
        # respect an explicit map_location; only supply one when absent
        if kwargs.get("map_location") is None:
            kwargs["map_location"] = torch.device("cpu")
        return _orig_load(*args, **kwargs)

    torch.load = _load
    applied.append("torch.load -> map_location=cpu when unspecified")

    # --- 7. default tensor type ---------------------------------------------
    if hasattr(torch, "set_default_tensor_type"):
        _orig_sdtt = torch.set_default_tensor_type
        _ORIGINALS[("torch", "set_default_tensor_type")] = _orig_sdtt

        def _sdtt(t):
            if isinstance(t, str) and "cuda" in t:
                t = t.replace("torch.cuda.", "torch.")
            return _orig_sdtt(t)

        torch.set_default_tensor_type = _sdtt

    _INSTALLED = True
    return {"already_installed": False, "applied": applied,
            "cuda_available": torch.cuda.is_available()}


def uninstall():
    """Restore everything. Mainly for tests."""
    global _INSTALLED
    if not _INSTALLED:
        return
    for (ns, name), original in _ORIGINALS.items():
        target = {"cuda": torch.cuda, "torch": torch,
                  "Tensor": torch.Tensor, "Module": torch.nn.Module}[ns]
        setattr(target, name, original)
    _ORIGINALS.clear()
    _INSTALLED = False


def selftest():
    """Verify the patch against the failure modes it exists to prevent."""
    results = []

    def ck(name, fn):
        try:
            fn()
            results.append((True, name, ""))
        except Exception as e:
            results.append((False, name, f"{type(e).__name__}: {e}"))

    # the exact crash from blocks.py line 68
    ck("torch.cuda.IntTensor([1,2,3])",
       lambda: torch.cuda.IntTensor([1, 2, 3]))
    ck("torch.cuda.IntTensor(5) size form",
       lambda: torch.cuda.IntTensor(5))
    ck("torch.cuda.FloatTensor(2,3).zero_()",
       lambda: torch.cuda.FloatTensor(2, 3).zero_())
    ck("torch.cuda.LongTensor([1])", lambda: torch.cuda.LongTensor([1]))
    ck("tensor.cuda()", lambda: torch.zeros(3).cuda())
    ck("tensor.to('cuda')", lambda: torch.zeros(3).to("cuda"))
    ck("tensor.to('cuda:0')", lambda: torch.zeros(3).to("cuda:0"))
    ck("tensor.to(torch.device('cuda'))",
       lambda: torch.zeros(3).to(torch.device("cuda")))
    ck("module.cuda()", lambda: torch.nn.Linear(2, 2).cuda())
    ck("module.to('cuda')", lambda: torch.nn.Linear(2, 2).to("cuda"))
    ck("torch.zeros(3, device='cuda')",
       lambda: torch.zeros(3, device="cuda"))
    ck("torch.tensor([1.], device='cuda')",
       lambda: torch.tensor([1.0], device="cuda"))
    ck("torch.cuda.empty_cache()", lambda: torch.cuda.empty_cache())
    ck("torch.cuda.synchronize()", lambda: torch.cuda.synchronize())

    # dtype and value must be preserved, not just 'no exception'
    def dtype_preserved():
        t = torch.cuda.IntTensor([7, 8])
        assert t.dtype == torch.int32, t.dtype
        assert t.tolist() == [7, 8], t.tolist()
        assert t.device.type == "cpu", t.device
    ck("dtype/values/device preserved", dtype_preserved)

    # .to(dtype) must NOT be hijacked by the device rewriting
    def dtype_to_passthrough():
        t = torch.zeros(3).to(torch.float64)
        assert t.dtype == torch.float64, t.dtype
    ck(".to(dtype) unaffected", dtype_to_passthrough)

    def tensor_to_passthrough():
        ref = torch.zeros(3, dtype=torch.float64)
        t = torch.zeros(3).to(ref)
        assert t.dtype == torch.float64, t.dtype
    ck(".to(other_tensor) unaffected", tensor_to_passthrough)

    return results


if __name__ == "__main__":
    print(install())
    print()
    ok = True
    for passed, name, detail in selftest():
        print(("  PASS  " if passed else "  FAIL  ") + name +
              (f"   {detail}" if detail else ""))
        ok &= passed
    print()
    print("all good" if ok else "FAILURES PRESENT")
