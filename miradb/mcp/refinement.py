"""Ontological refinement for model comparison.

Comparing concepts "by refinement" asks whether one grounding is an
ontological child of another (e.g. a specific infected state of a generic
one). MIRA answers this with a transitive closure of the DKG's
subclass/part-of relations, fetched from the DKG REST API. The closure is
large (~650k pairs, ~20 s to fetch), so it is loaded once, in the
background, and cached on disk.

The DKG URL is taken from ``MIRA_REST_URL`` or the ``mira``/``rest_url``
pystow config, falling back to the public epidemiology DKG.
"""

import gzip
import json
import logging
import os
import threading
import time
from typing import Callable, Literal

import pystow

logger = logging.getLogger(__name__)

DEFAULT_DKG_URL = (
    "http://mira-epi-dkg-lb-dc1e19b273dedaa2.elb.us-east-1.amazonaws.com/api"
)
CACHE_MAX_AGE_S = 7 * 24 * 3600
CACHE_PATH = pystow.join("miradb", "mcp", name="dkg_transitive_closure.json.gz")

RefinementMode = Literal["dkg", "exact"]


def _exact(child_curie: str, parent_curie: str) -> bool:
    """Refinement function under which no grounding refines another, so
    only equal concepts match."""
    return False


def dkg_url() -> str:
    return (
        os.environ.get("MIRA_REST_URL")
        or pystow.get_config("mira", "rest_url")
        or DEFAULT_DKG_URL
    )


class DkgClosure:
    """Lazily loaded DKG transitive closure, shared by all tools."""

    def __init__(self):
        self._closure: set[tuple[str, str]] | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def _read_cache(self) -> set[tuple[str, str]] | None:
        if not CACHE_PATH.exists():
            return None
        if time.time() - CACHE_PATH.stat().st_mtime > CACHE_MAX_AGE_S:
            return None
        try:
            with gzip.open(CACHE_PATH, "rt") as fh:
                return {tuple(pair) for pair in json.load(fh)}
        except (OSError, ValueError):
            logger.warning("Ignoring unreadable DKG closure cache %s", CACHE_PATH)
            return None

    def _write_cache(self, closure: set[tuple[str, str]]):
        try:
            with gzip.open(CACHE_PATH, "wt") as fh:
                json.dump(sorted(closure), fh)
        except OSError:
            logger.warning("Could not write DKG closure cache %s", CACHE_PATH)

    def load(self) -> set[tuple[str, str]] | None:
        """Return the closure, loading it on first use; None if the DKG is
        unreachable."""
        with self._lock:
            if self._closure is not None:
                return self._closure
            closure = self._read_cache()
            if closure is None:
                from mira.dkg.web_client import get_transitive_closure_web
                try:
                    closure = get_transitive_closure_web(api_url=dkg_url())
                except Exception as err:
                    # Don't cache the failure: the next call retries
                    self._error = f"{type(err).__name__}: {err}"
                    logger.warning("DKG transitive closure unavailable: %s",
                                   self._error)
                    return None
                self._write_cache(closure)
            self._closure = closure
            self._error = None
            logger.info("Loaded DKG transitive closure (%d pairs)", len(closure))
            return closure

    def prefetch(self):
        """Start loading the closure in a background thread."""
        threading.Thread(target=self.load, name="dkg-closure", daemon=True).start()

    @property
    def error(self) -> str | None:
        return self._error


dkg_closure = DkgClosure()


def get_refinement_func(
    mode: RefinementMode = "dkg",
) -> tuple[Callable[[str, str], bool], RefinementMode, str | None]:
    """Return a refinement function for ``mode``.

    Returns
    -------
    :
        The function, the mode actually used and a warning. When the DKG is
        unreachable, the mode falls back to ``exact`` and the warning says
        why, so results never silently claim ontology-aware matching.
    """
    if mode == "exact":
        return _exact, "exact", None
    closure = dkg_closure.load()
    if closure is None:
        return _exact, "exact", (
            f"DKG unavailable ({dkg_closure.error}); compared by exact "
            "grounding only."
        )
    return (lambda child, parent: (child, parent) in closure), "dkg", None
