"""HDF5 array-list storage helpers.

TL;DR
-----
This module saves and loads lists of NumPy arrays with variable shapes in one HDF5 file.

Notes
-----
It stores each array under a key, records metadata, and provides simple listing and iteration helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Union

import json
import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("h5py is required: pip install h5py") from e


ArrayLike = Union[np.ndarray, Sequence[Any]]


@dataclass(frozen=True)
class H5ArrayListSpec:
    """Naming spec for datasets in the group.
    
    TL;DR
    -----
    Naming spec for datasets in the group.
    """
    group: str = "solutions"
    dataset_prefix: str = "sol"
    digits: int = 6

    def name(self, i: int) -> str:
        """Return the dataset or group name stored in this specification.
        
        TL;DR
        -----
        Return the dataset or group name stored in this specification.
        
        Parameters
        ----------
        i : object
            Value supplied as `i` for this helper.
        
        Returns
        -------
        object
            Value produced by the helper.
        
        Notes
        -----
        The property gives callers a stable label for HDF5 storage.
        """
        return f"{self.dataset_prefix}_{i:0{self.digits}d}"


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-formatted string.
    
    TL;DR
    -----
    Return the current UTC time as an ISO-formatted string.
    
    Returns
    -------
    object
        Value produced by the helper.
    
    Notes
    -----
    This helper is part of the surrounding workflow and keeps behavior local to the caller.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_array_list_h5(
    filepath: Union[str, "Path"],
    arrays: Sequence[ArrayLike],
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    chunks: Union[bool, tuple, None] = True,
    overwrite: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a list of arrays (variable shapes allowed) into a single HDF5 file.
    
    TL;DR
    -----
    Save a list of arrays (variable shapes allowed) into a single HDF5 file.
    
    Parameters
    ----------
    filepath
        Target .h5/.hdf5 path.
    arrays
        Sequence of array-like objects. Each entry is converted by np.asarray.
    spec
        Group name and dataset naming scheme.
    compression
        HDF5 compression filter (e.g., "gzip", "lzf", None).
    compression_opts
        Compression level for gzip (0-9). Ignored for None and some filters.
    chunks
        Use chunking (True/False) or pass an explicit chunk shape tuple.
        For compression, chunking must be enabled.
    overwrite
        If True, recreate the file. If False, update/replace the target group.
    metadata
        Optional dict stored as JSON in group attrs ("metadata_json").
    
    Notes
    -----
    - Do not load untrusted HDF5 files if downstream workflow code executes based on
      stored metadata. The array payload itself is data-only.
    """
    filepath = str(filepath)
    mode = "w" if overwrite else "a"

    with h5py.File(filepath, mode) as f:
        # Replace group if present
        if spec.group in f:
            del f[spec.group]
        g = f.create_group(spec.group)

        # Group attrs
        g.attrs["n_items"] = int(len(arrays))
        g.attrs["dataset_prefix"] = spec.dataset_prefix
        g.attrs["digits"] = int(spec.digits)
        g.attrs["created_utc"] = _utc_now_iso()
        if metadata is not None:
            g.attrs["metadata_json"] = json.dumps(metadata)

        for i, item in enumerate(arrays):
            arr = np.asarray(item)
            dset_name = spec.name(i)

            create_kwargs = dict(
                data=arr,
                compression=compression,
                chunks=chunks,
            )
            if compression is not None and compression == "gzip":
                create_kwargs["compression_opts"] = compression_opts

            g.create_dataset(dset_name, **create_kwargs)


def list_keys_h5(
    filepath: Union[str, "Path"],
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
) -> List[str]:
    """Return dataset keys under the target group.
    
    TL;DR
    -----
    Return dataset keys under the target group.
    """
    filepath = str(filepath)
    with h5py.File(filepath, "r") as f:
        if spec.group not in f:
            raise KeyError(f"Group not found: {spec.group!r}")
        g = f[spec.group]
        return list(g.keys())


def load_array_h5(
    filepath: Union[str, "Path"],
    index: int,
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
) -> np.ndarray:
    """Load a single array by index.
    
    TL;DR
    -----
    Load a single array by index.
    """
    filepath = str(filepath)
    with h5py.File(filepath, "r") as f:
        if spec.group not in f:
            raise KeyError(f"Group not found: {spec.group!r}")
        g = f[spec.group]
        key = spec.name(index)
        if key not in g:
            raise KeyError(f"Dataset not found: {key!r}")
        return g[key][...]


def iter_array_list_h5(
    filepath: Union[str, "Path"],
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
) -> Iterator[np.ndarray]:
    """Iterate arrays one-by-one (reduces peak RAM versus loading all at once).
    
    TL;DR
    -----
    Iterate arrays one-by-one (reduces peak RAM versus loading all at once).
    """
    filepath = str(filepath)
    with h5py.File(filepath, "r") as f:
        if spec.group not in f:
            raise KeyError(f"Group not found: {spec.group!r}")
        g = f[spec.group]

        n = int(g.attrs.get("n_items", len(g.keys())))
        for i in range(n):
            key = spec.name(i)
            if key in g:
                yield g[key][...]
            else:
                # Fallback: iterate sorted keys if indices are not contiguous
                break
        else:
            return

        # Non-contiguous keys: fall back to lexical sort
        for key in sorted(g.keys()):
            yield g[key][...]


def load_array_list_h5(
    filepath: Union[str, "Path"],
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
) -> List[np.ndarray]:
    """Load the full list into memory.
    
    TL;DR
    -----
    Load the full list into memory.
    """
    return list(iter_array_list_h5(filepath, spec=spec))


def load_metadata_h5(
    filepath: Union[str, "Path"],
    *,
    spec: H5ArrayListSpec = H5ArrayListSpec(),
) -> Dict[str, Any]:
    """Load optional metadata dict stored at save time; returns {} if absent.
    
    TL;DR
    -----
    Load optional metadata dict stored at save time; returns {} if absent.
    """
    filepath = str(filepath)
    with h5py.File(filepath, "r") as f:
        if spec.group not in f:
            raise KeyError(f"Group not found: {spec.group!r}")
        g = f[spec.group]
        md_json = g.attrs.get("metadata_json", None)
        if md_json is None:
            return {}
        if isinstance(md_json, (bytes, bytearray)):
            md_json = md_json.decode("utf-8")
        return json.loads(md_json)
