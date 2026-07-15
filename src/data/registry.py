"""
src/data/registry.py — Dataset Registry for KalimCoder.

Responsibilities
----------------
1. Load ``configs/datasets.yaml`` (auto-resolved relative to project root).
2. Validate each entry against the ``DatasetEntry`` schema.
3. Expose :func:`get_enabled_datasets` which returns only enabled entries.

Typical usage
-------------
>>> from src.data.registry import get_enabled_datasets
>>> for ds in get_enabled_datasets():
...     print(ds.name, ds.repo_id)

Nothing is downloaded here — this module is pure metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
# Project root is two levels above this file (src/data/registry.py → project/).
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG: Path = _PROJECT_ROOT / "configs" / "datasets.yaml"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetEntry:
    """Immutable descriptor for a single dataset in the registry.

    Attributes
    ----------
    name:
        Unique snake_case identifier used across the pipeline.
    repo_id:
        Hugging Face Hub repository identifier (``org/name``) or a canonical
        dataset name recognised by ``datasets.load_dataset``.
    config:
        Optional dataset configuration / subset name forwarded directly to
        ``load_dataset(..., name=config)``. ``None`` means "use the default".
    split:
        Dataset split to consume (e.g. ``"train"``, ``"test"``).
    destination:
        Relative path (from project root) where the dataset will be cached on
        disk once the download pipeline runs.  Not created by this module.
    enabled:
        When ``False`` the entry is present in YAML but excluded from
        :func:`get_enabled_datasets`.
    """

    name: str
    repo_id: str
    split: str
    destination: str
    enabled: bool
    config: str | None = field(default=None)
    adapter: str | None = field(default=None)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def destination_path(self) -> Path:
        """Absolute :class:`~pathlib.Path` to the local destination folder."""
        return _PROJECT_ROOT / self.destination

    def __str__(self) -> str:  # pragma: no cover
        cfg = f"/{self.config}" if self.config else ""
        status = "on" if self.enabled else "off"
        adapter = f" adapter={self.adapter!r}" if self.adapter else ""
        return f"[{status}] {self.name} ({self.repo_id}{cfg}, split={self.split}{adapter})"


# ---------------------------------------------------------------------------
# Required and optional fields with their expected Python types
# ---------------------------------------------------------------------------
_REQUIRED_FIELDS: dict[str, type] = {
    "name": str,
    "repo_id": str,
    "split": str,
    "destination": str,
    "enabled": bool,
}
_OPTIONAL_FIELDS: dict[str, type | tuple[type, ...]] = {
    "config":  (str, type(None)),
    "adapter": (str, type(None)),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_entry(raw: Any, index: int) -> dict[str, Any]:
    """Validate a raw YAML mapping against the dataset schema.

    Parameters
    ----------
    raw:
        The parsed YAML object for one dataset entry.
    index:
        Zero-based position in the ``datasets`` list (used in error messages).

    Returns
    -------
    dict[str, Any]
        The validated entry as a plain dict, ready to be unpacked into
        :class:`DatasetEntry`.

    Raises
    ------
    TypeError
        If *raw* is not a ``dict``.
    ValueError
        If a required field is missing or a field has the wrong type.
    """
    if not isinstance(raw, dict):
        raise TypeError(
            f"datasets[{index}]: expected a mapping, got {type(raw).__name__!r}."
        )

    # Check required fields
    for field_name, expected_type in _REQUIRED_FIELDS.items():
        if field_name not in raw:
            raise ValueError(
                f"datasets[{index}]: missing required field {field_name!r}."
            )
        value = raw[field_name]
        if not isinstance(value, expected_type):
            raise ValueError(
                f"datasets[{index}].{field_name}: expected {expected_type.__name__}, "
                f"got {type(value).__name__!r} (value={value!r})."
            )

    # Check optional fields (type check only when present)
    for field_name, expected_types in _OPTIONAL_FIELDS.items():
        if field_name in raw:
            value = raw[field_name]
            if not isinstance(value, expected_types):
                type_names = (
                    " | ".join(t.__name__ for t in expected_types)
                    if isinstance(expected_types, tuple)
                    else expected_types.__name__
                )
                raise ValueError(
                    f"datasets[{index}].{field_name}: expected {type_names}, "
                    f"got {type(value).__name__!r} (value={value!r})."
                )

    # Warn about unrecognised keys (non-fatal)
    known_keys = set(_REQUIRED_FIELDS) | set(_OPTIONAL_FIELDS)
    unknown = set(raw) - known_keys
    if unknown:
        logger.warning(
            "datasets[%d] (%s): unknown keys ignored: %s",
            index,
            raw.get("name", "<unnamed>"),
            sorted(unknown),
        )

    return raw


def _load_yaml(config_path: Path) -> list[DatasetEntry]:
    """Parse ``config_path``, validate every entry, and return a list of
    :class:`DatasetEntry` objects.

    Raises
    ------
    FileNotFoundError
        If *config_path* does not exist.
    ValueError
        If the YAML top-level is not a dict with a ``datasets`` key, or if any
        entry fails schema validation.
    yaml.YAMLError
        If the file is not valid YAML.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Dataset registry not found: {config_path}\n"
            "Expected location: configs/datasets.yaml (project root)."
        )

    logger.debug("Loading dataset registry from %s", config_path)

    with config_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    if not isinstance(data, dict):
        raise ValueError(
            f"configs/datasets.yaml: top-level must be a mapping, "
            f"got {type(data).__name__!r}."
        )

    raw_list = data.get("datasets")
    if raw_list is None:
        raise ValueError(
            "configs/datasets.yaml: missing top-level key 'datasets'."
        )
    if not isinstance(raw_list, list):
        raise ValueError(
            f"configs/datasets.yaml: 'datasets' must be a list, "
            f"got {type(raw_list).__name__!r}."
        )
    if len(raw_list) == 0:
        logger.warning("configs/datasets.yaml: 'datasets' list is empty.")
        return []

    entries: list[DatasetEntry] = []
    for i, raw in enumerate(raw_list):
        validated = _validate_entry(raw, i)
        entry = DatasetEntry(
            name=validated["name"],
            repo_id=validated["repo_id"],
            config=validated.get("config"),       # optional
            split=validated["split"],
            destination=validated["destination"],
            adapter=validated.get("adapter"),     # optional
            enabled=validated["enabled"],
        )
        entries.append(entry)
        logger.debug("Registered dataset: %s", entry)

    logger.info(
        "Registry loaded: %d total dataset(s) from %s",
        len(entries),
        config_path,
    )
    return entries


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_registry(
    config_path: Path | str | None = None,
) -> list[DatasetEntry]:
    """Load and validate the full dataset registry (enabled *and* disabled).

    Parameters
    ----------
    config_path:
        Path to the YAML file. Defaults to ``configs/datasets.yaml`` resolved
        relative to the project root.

    Returns
    -------
    list[DatasetEntry]
        All dataset entries regardless of their ``enabled`` flag.

    Raises
    ------
    FileNotFoundError
        If the registry file cannot be found.
    ValueError
        If the YAML fails schema validation.
    """
    resolved = Path(config_path) if config_path is not None else _DEFAULT_CONFIG
    return _load_yaml(resolved)


def get_enabled_datasets(
    config_path: Path | str | None = None,
) -> list[DatasetEntry]:
    """Return only the dataset entries where ``enabled: true``.

    This is the primary entry-point for pipeline code that needs to iterate
    over active datasets without caring about disabled ones.

    Parameters
    ----------
    config_path:
        Optional override for the registry YAML path.  Falls back to
        ``configs/datasets.yaml`` at the project root when omitted.

    Returns
    -------
    list[DatasetEntry]
        Enabled datasets in the order they appear in the YAML file.

    Examples
    --------
    >>> from src.data.registry import get_enabled_datasets
    >>> datasets = get_enabled_datasets()
    >>> for ds in datasets:
    ...     print(ds.name, "→", ds.repo_id)

    Notes
    -----
    The function is deliberately *not* cached so that tests can swap the YAML
    path and callers always observe the current state of the file.
    """
    all_entries = load_registry(config_path)
    enabled = [e for e in all_entries if e.enabled]

    disabled_names = [e.name for e in all_entries if not e.enabled]
    if disabled_names:
        logger.info("Disabled datasets (skipped): %s", disabled_names)

    logger.info(
        "%d of %d dataset(s) enabled: %s",
        len(enabled),
        len(all_entries),
        [e.name for e in enabled],
    )
    return enabled
