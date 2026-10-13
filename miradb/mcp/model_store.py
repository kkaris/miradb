"""Registry of the TemplateModels an agent works with.

Template models are large, so tools pass them around by handle rather than
as JSON. A handle (``model_ref``) is one of:

- ``db:<id>``: a row of ``mira_template_models``, by its ``id``
- ``ws:<hex>``: a workspace model created during this server's lifetime,
  e.g. by stratifying or re-parameterizing another model

Every ``get`` returns a fresh copy, so tools are free to mutate the model
they receive.
"""

import json
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Callable

from sqlalchemy import select

from mira.metamodel import TemplateModel

from miradb.db.client import MiraDatabaseClient
from miradb.db.schema import MiraModel

logger = logging.getLogger(__name__)

DB_PREFIX = "db:"
WORKSPACE_PREFIX = "ws:"


class ModelNotFoundError(LookupError):
    pass


class ModelStore:
    """Resolve model handles to TemplateModels and hold workspace models.

    Parameters
    ----------
    client_getter :
        A callable returning the MiraDatabaseClient used to load ``db:``
        models. Called lazily so the store can be created before the
        database is reachable.
    max_workspace_models :
        Maximum number of workspace models kept; the oldest are evicted
        first. Default: 500.
    max_cached_db_models :
        Maximum number of database model JSONs cached. Default: 200.
    """

    def __init__(
        self,
        client_getter: Callable[[], MiraDatabaseClient],
        *,
        max_workspace_models: int = 500,
        max_cached_db_models: int = 200,
    ):
        self._client_getter = client_getter
        self._max_workspace = max_workspace_models
        self._max_db_cache = max_cached_db_models
        # ref -> {"json": dict, "parent": str | None, "operation": str | None}
        self._workspace: OrderedDict[str, dict] = OrderedDict()
        self._db_cache: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def normalize_ref(ref: str | int) -> str:
        """Return the canonical form of a handle.

        A bare integer (or digit string) is taken to be a
        ``mira_template_models.id``.
        """
        if isinstance(ref, int):
            return f"{DB_PREFIX}{ref}"
        ref = str(ref).strip()
        if ref.isdigit():
            return f"{DB_PREFIX}{ref}"
        if ref.startswith((DB_PREFIX, WORKSPACE_PREFIX)):
            return ref
        raise ModelNotFoundError(
            f"Invalid model_ref {ref!r}: expected 'db:<id>' or 'ws:<id>'."
        )

    def get_json(self, ref: str | int) -> dict:
        """Return the TemplateModel JSON of a handle."""
        ref = self.normalize_ref(ref)
        with self._lock:
            if ref.startswith(WORKSPACE_PREFIX):
                entry = self._workspace.get(ref)
                if entry is None:
                    raise ModelNotFoundError(
                        f"No workspace model {ref!r}. Workspace models do "
                        "not survive a server restart."
                    )
                return entry["json"]
            if ref in self._db_cache:
                self._db_cache.move_to_end(ref)
                return self._db_cache[ref]

        model_id = ref[len(DB_PREFIX):]
        if not model_id.isdigit():
            raise ModelNotFoundError(f"Invalid model_ref {ref!r}.")
        row = self._client_getter().query_one(
            select(MiraModel.mira_template_model).where(
                MiraModel.id == int(model_id)
            )
        )
        if not row or not row["mira_template_model"]:
            raise ModelNotFoundError(f"No template model with id {model_id}.")
        raw = row["mira_template_model"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        with self._lock:
            self._db_cache[ref] = raw
            while len(self._db_cache) > self._max_db_cache:
                self._db_cache.popitem(last=False)
        return raw

    def get(self, ref: str | int) -> TemplateModel:
        """Return a fresh TemplateModel for a handle."""
        return TemplateModel.from_json(self.get_json(ref))

    def add(
        self,
        model: TemplateModel,
        *,
        parent: str | None = None,
        operation: str | None = None,
    ) -> str:
        """Store a model in the workspace and return its new handle."""
        ref = f"{WORKSPACE_PREFIX}{uuid.uuid4().hex[:12]}"
        entry = {
            "json": model.to_json(),
            "parent": self.normalize_ref(parent) if parent else None,
            "operation": operation,
        }
        with self._lock:
            self._workspace[ref] = entry
            while len(self._workspace) > self._max_workspace:
                evicted, _ = self._workspace.popitem(last=False)
                logger.info("Evicted workspace model %s", evicted)
        return ref

    def lineage(self, ref: str | int) -> list[dict]:
        """Return the chain of operations that produced a handle.

        The list starts at ``ref`` and walks back to the original model.
        """
        ref = self.normalize_ref(ref)
        chain = []
        with self._lock:
            while ref:
                entry = self._workspace.get(ref)
                if entry is None:
                    chain.append({"model_ref": ref, "operation": None})
                    break
                chain.append({"model_ref": ref, "operation": entry["operation"]})
                ref = entry["parent"]
        return chain

    def list_workspace(self) -> list[dict]:
        """Return the handles of all workspace models, oldest first."""
        with self._lock:
            return [
                {
                    "model_ref": ref,
                    "parent": entry["parent"],
                    "operation": entry["operation"],
                }
                for ref, entry in self._workspace.items()
            ]
