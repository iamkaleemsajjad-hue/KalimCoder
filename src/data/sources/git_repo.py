"""
src/data/sources/git_repo.py — Stub for future GitHub repository ingestion.

This stub exists to establish the ``DatasetSource`` extension point for
repository-level code ingestion.  It will be implemented when KalimCoder
begins ingesting raw GitHub repositories (e.g. via the GitHub API, GH Archive,
or a local clone of BigCode's ``the-stack`` crawl pipeline).

Planned functionality
---------------------
* Clone or fetch a repository by URL.
* Walk all source files matching configured extensions.
* Extract file content, infer language from extension, resolve licence from
  SPDX tags or ``LICENSE`` files.
* Yield :class:`~src.data.schema.CanonicalExample` objects with
  ``task_type="completion"`` and full repository provenance.
"""

from __future__ import annotations

from typing import Generator, Iterator

from src.data.schema import CanonicalExample
from src.data.sources.base import DatasetSource


class GitRepositorySource(DatasetSource):
    """Stub: GitHub repository ingestion source.

    Parameters
    ----------
    repo_url:
        HTTPS URL of the repository to ingest.
    dataset_name:
        Registry name to assign in :class:`~src.data.schema.CanonicalExample`.
    extensions:
        File extensions to include (e.g. ``[".py", ".js"]``).
    license:
        SPDX identifier for the repository licence.

    Raises
    ------
    NotImplementedError
        Always — this source is not yet implemented.
    """

    def __init__(
        self,
        repo_url: str,
        dataset_name: str = "git_repository",
        extensions: list[str] | None = None,
        license: str = "unknown",
    ) -> None:
        self._repo_url = repo_url
        self._dataset_name = dataset_name
        self._extensions = extensions or [".py", ".js", ".ts", ".java", ".go", ".rs"]
        self._license = license

    @property
    def name(self) -> str:
        return f"GitRepositorySource({self._repo_url})"

    @property
    def estimated_rows(self) -> int | None:
        return None  # unknown until cloned

    @property
    def supports_streaming(self) -> bool:
        return False  # requires local clone first

    def iter_raw_rows(self) -> Iterator[dict]:
        raise NotImplementedError(
            "GitRepositorySource is not yet implemented. "
            "Use HuggingFaceSource, LocalArrowSource, JSONLSource, or ParquetSource."
        )

    def iter_canonical_rows(self) -> Generator[CanonicalExample, None, None]:
        raise NotImplementedError(
            "GitRepositorySource is not yet implemented."
        )
        # Unreachable; satisfies generator type
        yield  # type: ignore[misc]
