"""Shared fixtures for energy-scheduling-benchmark tests."""

from __future__ import annotations

import importlib
import sys

# ---------------------------------------------------------------------------
# Fix mango import resolution when pytest is run from the repo root.
#
# The repo root contains a bare `mango/` directory (the local editable
# source).  When pytest's CWD is the repo root, Python's PathFinder discovers
# `mango/` as a namespace package (no __init__.py at the top level) BEFORE
# the editable-install finder gets a chance to map it to the real inner
# `mango/mango/` package.  Moving the editable finder ahead of PathFinder
# ensures the correct package is loaded regardless of CWD.
# ---------------------------------------------------------------------------
_path_finder_idx = next(
    (i for i, f in enumerate(sys.meta_path) if getattr(f, "__name__", "") == "PathFinder"),
    None,
)
for _ef in list(sys.meta_path):
    if getattr(_ef, "__name__", "") != "_EditableFinder":
        continue
    _mod = importlib.import_module(_ef.__module__)
    if "mango" not in getattr(_mod, "MAPPING", {}):
        continue
    sys.meta_path.remove(_ef)
    sys.meta_path.insert(_path_finder_idx if _path_finder_idx is not None else 0, _ef)
    break


