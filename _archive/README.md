# _archive — dead code, kept for reference only

**Nothing in this directory is part of the running application.** It is not imported, not launched,
and not tested. It is kept because some of it records how the project got here.

Everything here is also in git history from the baseline commit, so it can be recovered even if this
directory is deleted.

| File | Why it is dead |
|---|---|
| `server.py` | An older FastAPI router layer. It describes itself as "thin routing over api_core" and **no longer matches it** — 10 stale routes against the live 14. Nothing launches it: `start_backend.bat` runs `uvicorn api_core:app`. The live API is `api_core.py`. |
| `test_stl_parser.py` | Tests the STL parser *inside* `server.py`, loading it by path with a stubbed `fastapi` module. The live parser is `stl_io.parse_stl_bytes`, and `test_face_order.py` covers it. Was never wired into `run_all_tests.py`. |
| `test_api.py` | A 15-line throwaway that POSTs a **hardcoded absolute path to a real patient scan** to a running server. Never wired into the runner. See the PHI note below. |
| `frontend_App.jsx` | A loose root-level copy of the client from 2026-08-28, superseded by `frontend/src/App.jsx`. |
| `frontend_src/App.jsx` | A 192-line stub against the live 1,526-line `frontend/src/App.jsx`. The most dangerous file in the repo for an automated reader: same filename, plausible content, completely obsolete. |
| `tgn_diagnose_model.py.py` | Duplicate of `tgn_diagnose_model.py` with a doubled extension. The single-extension file is live. |
| `config.py` | **0 bytes**, and it *shadowed* `tooth_segmentation/config.py` — `import config` from the repo root resolved here and yielded zero symbols. |
| `models.py` | **0 bytes**, same shadowing problem against `tooth_segmentation/models.py`. Verified before moving: nothing in the app or the vendored tree does a bare `import config` / `import models`, so removing them from the import path is safe and closes a latent trap. |
| `HANDOVER.md` | Its architecture section is **factually false**: it calls `server.py` "FastAPI routes over api_core" and `api_core.py` "backend logic (no FastAPI import)". Both were true once and neither is now. Superseded by `CLAUDE.md` (invariants, measured behaviour) and `MASTER_APP_HANDOVER.md` (file map, current state). Kept because §4 §5 §6 of it still contain real measurements that were folded forward. |

## PHI note on `test_api.py`

It contains a patient-identifying filename in a hardcoded path
(`...\17012026-<name>-lowerjaw.stl`). The name appears to be the repository owner's own. It is a
string in a dead script, not a stored scan, and the referenced file is gitignored — but it **was
captured in the baseline commit before this archive was created**, so it exists in git history.

No action was taken on history without asking. If you want it gone, the options are to rewrite
history (`git filter-repo`) while the repo has no remote and no collaborators — which is the cheapest
it will ever be — or to accept it as your own name in a local-only repository. Flagging it rather
than deciding for you.

## What was deliberately NOT archived

- `app_ui.py` — the PyQt6/PyVista desktop shell. Legacy relative to the web product, but it still
  runs and still exercises `core_geometry`, and `test_ui_dataflow.py` / `test_worker_path.py` cover
  its data path.
- `test_picking.py`, `test_picking2.py` — VTK click-picking experiments for `app_ui.py`. Orphaned
  from the runner, but they target a path that still exists.
- `test_bleed_diagnosis.py` — a characterization script against live `core_geometry`. Orphaned from
  the runner but not dead.

These four are orphans, not corpses. They are listed in `README.md` under development scripts.
