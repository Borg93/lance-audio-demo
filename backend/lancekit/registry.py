"""Lazy multi-dataset registry — one serving app, many Lance DBs.

Datasets are directories named ``<id>.lance`` under ``MEDIA_DB_ROOT``. Each is
opened on first use and cached with its validated
:class:`~backend.lancekit.descriptor.DatasetDescriptor`; an invalid or missing
descriptor surfaces as a domain error, never a half-served dataset. The
legacy single-DB routes resolve through :meth:`default`, so the pre-split API
surface keeps working while new routes address datasets explicitly.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import lancedb
from pydantic import BaseModel, ConfigDict

from backend.lancekit.descriptor import DatasetDescriptor, load_dataset_descriptor

logger = logging.getLogger(__name__)


class DatasetHandle(BaseModel):
    """One opened dataset: its lancedb connection + merged descriptor."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    path: Path
    db: lancedb.DBConnection
    descriptor: DatasetDescriptor

    def refresh_descriptor(self, descriptor_dir: str | Path) -> None:
        self.descriptor = load_dataset_descriptor(self.path, self.id, descriptor_dir)


class UnknownDatasetError(LookupError):
    """No ``<id>.lance`` directory under the registry root."""


class DatasetRegistry:
    """Thread-safe lazy cache of :class:`DatasetHandle` by dataset id."""

    def __init__(self, root: str | Path, descriptor_dir: str | Path, default_id: str) -> None:
        self._root = Path(root)
        self._descriptor_dir = Path(descriptor_dir)
        self._default_id = default_id
        self._handles: dict[str, DatasetHandle] = {}
        self._lock = threading.Lock()

    def list_ids(self) -> list[str]:
        return sorted(p.stem for p in self._root.glob("*.lance") if p.is_dir())

    def get(self, dataset_id: str) -> DatasetHandle:
        # A dataset id is a single flat directory stem: reject anything with a
        # path separator or dot segment so `..`/`/`/`.` can't escape the root,
        # alias the default, or mint duplicate caches (audit: path traversal via
        # the `dataset` query param). Ids must match what list_ids() advertises.
        if dataset_id in {"", ".", ".."} or "/" in dataset_id or "\\" in dataset_id:
            raise UnknownDatasetError(f"invalid dataset id {dataset_id!r}")
        with self._lock:
            handle = self._handles.get(dataset_id)
            if handle is not None:
                return handle
            path = self._root / f"{dataset_id}.lance"
            if path.parent != self._root or not path.is_dir():
                raise UnknownDatasetError(f"dataset {dataset_id!r} not found under {self._root}")
            descriptor = load_dataset_descriptor(path, dataset_id, self._descriptor_dir)
            handle = DatasetHandle(
                id=dataset_id,
                path=path,
                db=lancedb.connect(str(path)),
                descriptor=descriptor,
            )
            self._handles[dataset_id] = handle
            logger.info("opened dataset %s (%d tables)", dataset_id, len(descriptor.tables))
            return handle

    def default(self) -> DatasetHandle:
        return self.get(self._default_id)
