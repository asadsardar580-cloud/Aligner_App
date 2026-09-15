"""
tgn_bridge.py — the single point of contact between the FastAPI backend and
the vendored ToothGroupNetwork checkout.

DESIGN RULES, and why each one exists
-------------------------------------

1. NOTHING IMPORTS AT MODULE LOAD.
   `import tgn_bridge` is cheap and cannot fail. The heavy import happens on
   first real use, inside a lock. If ToothGroupNetwork is broken, missing, or
   half-installed, the server still starts and every other endpoint — export,
   kinematics, socket carving — keeps working. Segmentation is an enhancement;
   it must never be able to take down the clinical loop. A top-level
   `from inference_pipelines... import ...` in api_core.py makes a broken
   checkout fatal to the whole application.

2. THE VENDORED REPO IS NEVER MODIFIED.
   Generating `__init__.py` files into a third-party checkout on every server
   boot mutates code you do not own, fails on a read-only or permission-
   restricted directory, and dirties `git status`. Since Python 3.3 a
   directory without `__init__.py` is importable as a namespace package, so
   the generation is usually unnecessary anyway. `diagnose()` reports what is
   missing; `materialise_packages()` writes them only if you explicitly ask.

3. sys.path ORDERING IS EXPLICIT.
   `sys.path.insert(0, TGN_PATH)` puts every top-level name in that repo —
   `models`, `runner`, `generator`, `trainer`, `predict_utils`, `ops_utils` —
   ahead of YOUR modules and ahead of site-packages, permanently, for the life
   of the process. `models` in particular is a name you are likely to want.
   This module gives the repo priority only during its own import (so its
   internal imports resolve to its own files) and then demotes it to the end
   of the path, where it can still satisfy lazy imports but cannot capture a
   name from your code.

4. FAILURES ARE STRUCTURED, NOT PRINTED.
   `status()` returns a dict an endpoint can serialise. A caught exception
   that only reaches stdout is invisible behind a web server.
"""

import os
import sys
import threading
import traceback

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

TGN_DIRNAME = "ToothGroupNetwork"

_LOCK = threading.Lock()
_STATE = {
    "loaded": False,
    "error": None,
    "traceback": None,
    "pipeline": None,
    "predictor": None,
    "checkpoints": {},
}


def tgn_path(base=None):
    """Absolute path to the vendored checkout."""
    base = base or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, TGN_DIRNAME)


class _RepoOnPath:
    """
    Context manager: give the vendored repo first claim on imports for the
    duration of the block, then leave it at the END of sys.path.

    Demotion rather than removal is deliberate. Removing it entirely would
    break any import the repo performs lazily inside a function at inference
    time — those would fail minutes later, far from this code, and look like a
    model bug. Leaving it last keeps it importable while ensuring a name
    collision resolves in favour of your own modules.
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        while self.path in sys.path:
            sys.path.remove(self.path)
        sys.path.insert(0, self.path)
        return self

    def __exit__(self, *exc):
        while self.path in sys.path:
            sys.path.remove(self.path)
        sys.path.append(self.path)
        return False


# --------------------------------------------------------------------------
# diagnostics — run these before blaming the model
# --------------------------------------------------------------------------

def diagnose(base=None):
    """
    Report the state of the checkout WITHOUT importing or modifying anything.
    Safe to expose on a debug endpoint.
    """
    root = tgn_path(base)
    out = {
        "tgn_path": root,
        "exists": os.path.isdir(root),
        "python": sys.version.split()[0],
        "expected_entry_points": {},
        "dirs_without_init": [],
        "compiled_extensions_found": [],
    }
    if not out["exists"]:
        out["hint"] = f"No directory at {root}. Check TGN_DIRNAME."
        return out

    for rel in ("inference_pipelines/inference_pipeline_maker.py",
                "predict_utils.py",
                "external_libs/pointops/functions/pointops.py",
                "external_libs/pointnet2_utils.py",
                "models"):
        out["expected_entry_points"][rel] = os.path.exists(
            os.path.join(root, rel))

    skip = {".git", ".vscode", "__pycache__", "ckpts", "build", "dist"}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip
                       and not d.endswith(".egg-info")]
        if any(f.endswith(".py") for f in filenames):
            if "__init__.py" not in filenames:
                out["dirs_without_init"].append(
                    os.path.relpath(dirpath, root))
        for f in filenames:
            if f.endswith((".so", ".pyd")):
                out["compiled_extensions_found"].append(
                    os.path.relpath(os.path.join(dirpath, f), root))

    # A missing __init__.py is usually harmless (namespace packages). A
    # missing entry-point file is not.
    out["hint"] = (
        "Missing __init__.py files are normally fine on Python 3.3+; only "
        "materialise them if an import genuinely fails. If "
        "inference_pipeline_maker.py is False, the checkout is incomplete.")
    return out


def materialise_packages(base=None, dry_run=True):
    """
    Write `__init__.py` into every package directory that lacks one.

    Opt-in and dry-run by default, because this edits a repository you do not
    own. Try `diagnose()` and a real import first; reach for this only if an
    import actually fails in a way namespace packages cannot satisfy.
    """
    root = tgn_path(base)
    skip = {".git", ".vscode", "__pycache__", "ckpts", "build", "dist"}
    written = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip
                       and not d.endswith(".egg-info")]
        if not any(f.endswith(".py") for f in filenames):
            continue
        target = os.path.join(dirpath, "__init__.py")
        if os.path.exists(target):
            continue
        written.append(os.path.relpath(target, root))
        if not dry_run:
            with open(target, "w") as f:
                f.write("# generated by tgn_bridge.materialise_packages\n")
    return {"dry_run": dry_run, "files": written}


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def _translate_illegal_raise(exc, model_name, ckpts, root):
    """
    Turn `TypeError: exceptions must derive from BaseException` into something
    that names the actual problem.

    That TypeError is what Python emits when code does `raise "some string"`,
    Python-2 style. The string — which is the real diagnostic — is DESTROYED in
    the process. In this repository the pattern sits in the final `else` of the
    model_name dispatch, so the message you never get to read is some variant of
    "not implemented model name".

    Note what this function does and does not do. It does not suppress the
    error or let loading continue. Swallowing this exception would build a
    DIFFERENT architecture from the one intended and then quietly mis-segment a
    patient's arch, which is far worse than refusing to start. It only restores
    the information the bad raise threw away.
    """
    if not (isinstance(exc, TypeError)
            and "must derive from BaseException" in str(exc)):
        return None

    names = []
    try:
        import tgn_diagnose_model as diag
        rep = diag.valid_model_names(root)
        names = rep.get("accepted", [])
    except Exception:
        pass

    msg = [
        f"The repository rejected model_name={model_name!r} using a "
        f"Python-2 style `raise \"string\"`, which Python 3 converts into "
        f"TypeError and discards the message.",
    ]
    if names:
        msg.append(f"Names this checkout actually accepts: {names}.")
        if model_name not in names:
            msg.append(f"{model_name!r} is NOT among them.")
    msg.append(
        f"Passed {len(ckpts)} checkpoint path(s). Passing the same file twice "
        f"is not a substitute for a second checkpoint: the FPS and BDL stages "
        f"are different networks, and a duplicate will mismatch on shapes.")
    msg.append("Run: python tgn_diagnose_model.py <tgn_root> <checkpoint>")
    return " ".join(msg)


def load(checkpoint_path, checkpoint_path_bdl=None, model_name="tgnet",
         base=None, force=False):
    """
    Import ToothGroupNetwork and build the inference pipeline. Idempotent and
    thread-safe: concurrent requests during startup will not import twice.

    Returns status(). Never raises — inspect the returned dict.
    """
    with _LOCK:
        if _STATE["loaded"] and not force:
            return status()

        _STATE.update(loaded=False, error=None, traceback=None,
                      pipeline=None, predictor=None)
        root = tgn_path(base)

        try:
            if not os.path.isdir(root):
                raise FileNotFoundError(f"ToothGroupNetwork not found at {root}")
            for p in (checkpoint_path, checkpoint_path_bdl):
                if p and not os.path.exists(p):
                    raise FileNotFoundError(f"checkpoint not found: {p}")

            # Device compatibility must be in place before any repo module is
            # imported: some of them bind torch.cuda attributes at import time.
            import cpu_compat
            cpu_compat.install()

            # The shim must be installed before any repo module is imported;
            # Python caches on first import and a late install leaves the
            # broken CUDA wrapper in place.
            import pointops_cpu
            pointops_cpu.install()

            with _RepoOnPath(root):
                from inference_pipelines.inference_pipeline_maker import (
                    make_inference_pipeline)
                from predict_utils import ScanSegmentation

                ckpts = [checkpoint_path]
                if checkpoint_path_bdl:
                    ckpts.append(checkpoint_path_bdl)
                # Deliberately NOT padded to length 2 by repeating the first
                # path. The FPS and BDL stages are separate networks with
                # separate weights; a duplicate satisfies a length check and
                # then fails on shapes, or worse, loads and produces a model
                # that is half wrong in a way nothing reports.
                pipeline = make_inference_pipeline(model_name, ckpts)
                predictor = ScanSegmentation(pipeline)

            _STATE.update(loaded=True, pipeline=pipeline, predictor=predictor,
                          checkpoints={"fps": checkpoint_path,
                                       "bdl": checkpoint_path_bdl},
                          model_name=model_name)
        except BaseException as exc:      # noqa: BLE001 — surfaced, not swallowed
            translated = _translate_illegal_raise(
                exc, model_name, [checkpoint_path] +
                ([checkpoint_path_bdl] if checkpoint_path_bdl else []), root)
            _STATE["error"] = translated or f"{type(exc).__name__}: {exc}"
            _STATE["traceback"] = traceback.format_exc()
        return status()


def status():
    """Serialisable state, safe to return from an endpoint."""
    return {
        "loaded": _STATE["loaded"],
        "error": _STATE["error"],
        "traceback": _STATE["traceback"],
        "checkpoints": _STATE["checkpoints"],
        "model_name": _STATE.get("model_name"),
        "device": "cpu",
    }


def predictor():
    """The ScanSegmentation instance, or None if loading failed."""
    return _STATE["predictor"]


if __name__ == "__main__":
    import json
    print(json.dumps(diagnose(), indent=2))
