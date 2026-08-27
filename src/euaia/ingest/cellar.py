"""Client for the EU Publications Office CELLAR repository.

Two capabilities:

* **Version resolution** via the SPARQL endpoint -- find every consolidated version of a
  regulation and pick the newest. This is the change-detection signal: consolidated acts
  are separate works with their own CELEX, so "is there a new version?" is a metadata
  question, not a diff of rendered HTML.
* **Content retrieval** via the REST API with ``Accept``-header content negotiation,
  preferring Formex XML (structured, article-level markup) over XHTML.

The predicate used for consolidation was established empirically against the live endpoint:
``cdm:act_consolidated_consolidates_resource_legal`` runs *from* the consolidated act *to*
the base act. Note that acts amended **by** the AI Act also carry this predicate pointing at
it, so the link alone over-matches; we additionally require the consolidated CELEX to derive
from the base CELEX (``32024R1689`` -> ``02024R1689-<date>``).

CELLAR asks API users to stay below 5 concurrent requests, back off on 429/503, and send a
descriptive User-Agent. All three are honoured here.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from euaia.config import settings

log = logging.getLogger(__name__)

CDM = "http://publications.europa.eu/ontology/cdm#"

# Content negotiation values that CELLAR actually honours (verified against the live API;
# `application/xml;mtype=fmx4` returns 400 -- Formex is only served zipped).
ACCEPT_FORMEX_ZIP = "application/zip;mtype=fmx4"
ACCEPT_XHTML = "application/xhtml+xml"


class CellarError(RuntimeError):
    """CELLAR returned something we cannot use."""


@dataclass(frozen=True, slots=True)
class VersionRef:
    """A retrievable version of a legal act."""

    cellar_id: str
    celex: str
    doc_date: dt.date | None
    eli: str | None = None
    is_consolidated: bool = False

    @property
    def label(self) -> str:
        """Human version label, e.g. 'consolidated 2026-07-27' or 'as adopted 2024-06-13'.

        The distinction is not cosmetic: a consolidated act carries amendments, an adopted
        act does not, and an answer's provenance record needs to say which it read.
        """
        if not self.doc_date:
            return self.celex
        kind = "consolidated" if self.is_consolidated else "as adopted"
        return f"{kind} {self.doc_date.isoformat()}"


@dataclass(frozen=True, slots=True)
class Manifestation:
    """Raw bytes retrieved for one version, plus how we got them."""

    content: bytes
    fmt: str  # "formex" | "xhtml"
    sha256: str
    filename: str | None = None


def consolidated_celex_prefix(base_celex: str) -> str:
    """``32024R1689`` -> ``02024R1689``.

    Consolidated acts replace the leading sector digit with ``0`` and append ``-<date>``.
    """
    if not base_celex:
        raise ValueError("base_celex must not be empty")
    return "0" + base_celex[1:]


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.TransportError)


class CellarClient:
    """Synchronous CELLAR client.

    Requests are issued sequentially, which keeps us inside CELLAR's concurrency guidance
    by construction; ``settings.cellar_max_concurrency`` documents the ceiling for any
    future parallel fetching.
    """

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(
            timeout=settings.cellar_timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": settings.cellar_user_agent},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CellarClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------ SPARQL

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def sparql(self, query: str) -> list[dict[str, str]]:
        """Run a SPARQL query, returning rows as flat ``{var: value}`` dicts."""
        resp = self._client.get(
            settings.cellar_sparql_endpoint,
            params={"query": query, "format": "application/sparql-results+json"},
            headers={"Accept": "application/sparql-results+json"},
        )
        if not _ok(resp):
            resp.raise_for_status()
        payload = resp.json()
        return [
            {var: binding["value"] for var, binding in row.items()}
            for row in payload["results"]["bindings"]
        ]

    def resolve_base_work(self, base_celex: str) -> VersionRef:
        """Find the cellar id of the original (as-adopted) act."""
        rows = self.sparql(
            f"""
            PREFIX cdm: <{CDM}>
            SELECT ?work ?date WHERE {{
              ?work cdm:resource_legal_id_celex ?celex .
              FILTER(STR(?celex) = "{base_celex}")
              OPTIONAL {{ ?work cdm:work_date_document ?date . }}
            }} LIMIT 1
            """
        )
        if not rows:
            raise CellarError(f"no CELLAR work found for CELEX {base_celex}")
        row = rows[0]
        return VersionRef(
            cellar_id=_cellar_id(row["work"]),
            celex=base_celex,
            doc_date=_parse_date(row.get("date")),
            is_consolidated=False,
        )

    def resolve_consolidated_versions(self, base_celex: str) -> list[VersionRef]:
        """Every consolidated version of ``base_celex``, newest first.

        Filtered on both the consolidation link *and* the derived CELEX prefix -- the link
        alone also matches consolidated versions of other acts that the base act amends.
        """
        prefix = consolidated_celex_prefix(base_celex)
        rows = self.sparql(
            f"""
            PREFIX cdm: <{CDM}>
            SELECT DISTINCT ?work ?celex ?date ?eli WHERE {{
              ?base cdm:resource_legal_id_celex ?baseCelex .
              FILTER(STR(?baseCelex) = "{base_celex}")
              ?work cdm:act_consolidated_consolidates_resource_legal ?base ;
                    cdm:resource_legal_id_celex ?celex .
              FILTER(STRSTARTS(STR(?celex), "{prefix}"))
              OPTIONAL {{ ?work cdm:work_date_document ?date . }}
              OPTIONAL {{ ?work cdm:resource_legal_eli ?eli . }}
            }}
            ORDER BY DESC(?date)
            """
        )
        versions = [
            VersionRef(
                cellar_id=_cellar_id(row["work"]),
                celex=row["celex"],
                doc_date=_parse_date(row.get("date")),
                eli=row.get("eli"),
                is_consolidated=True,
            )
            for row in rows
        ]
        # Sort defensively: SPARQL ordering over OPTIONAL dates is not guaranteed.
        versions.sort(key=lambda v: (v.doc_date or dt.date.min, v.celex), reverse=True)
        return versions

    def latest_consolidated(self, base_celex: str) -> VersionRef:
        versions = self.resolve_consolidated_versions(base_celex)
        if not versions:
            raise CellarError(f"no consolidated version found for CELEX {base_celex}")
        return versions[0]

    # ----------------------------------------------------------------- content

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TransportError)),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get_manifestation(self, cellar_id: str, accept: str, language: str) -> httpx.Response:
        resp = self._client.get(
            f"{settings.cellar_resource_base}/{cellar_id}",
            headers={"Accept": accept, "Accept-Language": language},
        )
        if not _ok(resp):
            resp.raise_for_status()
        return resp

    def fetch(self, cellar_id: str, language: str = "eng") -> Manifestation:
        """Retrieve a version's content, preferring Formex XML over XHTML.

        Formex gives explicit ARTICLE / PARAG / CONS.ANNEX markup, which is what makes
        article-level citation possible. XHTML is the fallback for documents CELLAR does
        not publish in Formex.
        """
        try:
            resp = self._get_manifestation(cellar_id, ACCEPT_FORMEX_ZIP, language)
            filename, xml = extract_formex_xml(resp.content)
            return Manifestation(
                content=xml,
                fmt="formex",
                sha256=hashlib.sha256(xml).hexdigest(),
                filename=filename,
            )
        except (httpx.HTTPStatusError, CellarError, zipfile.BadZipFile) as exc:
            log.warning(
                "Formex unavailable for %s (%s); falling back to XHTML", cellar_id, exc
            )

        resp = self._get_manifestation(cellar_id, ACCEPT_XHTML, language)
        return Manifestation(
            content=resp.content,
            fmt="xhtml",
            sha256=hashlib.sha256(resp.content).hexdigest(),
        )


def extract_formex_xml(zip_bytes: bytes) -> tuple[str, bytes]:
    """Pull the main document XML out of a Formex ZIP.

    A Formex bundle holds the act plus per-annex files and a small ``*.doc.xml`` metadata
    wrapper. The act itself is reliably the largest non-``.doc.xml`` entry (530 KB vs
    1.5 KB for the wrapper in the consolidated AI Act).
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        candidates = [
            info
            for info in zf.infolist()
            if info.filename.lower().endswith(".xml")
            and not info.filename.lower().endswith(".doc.xml")
        ]
        if not candidates:
            raise CellarError("Formex archive contains no usable XML entry")
        main = max(candidates, key=lambda i: i.file_size)
        return main.filename, zf.read(main.filename)


def _ok(resp: httpx.Response) -> bool:
    return 200 <= resp.status_code < 300


def _cellar_id(work_uri: str) -> str:
    """``http://publications.europa.eu/resource/cellar/<uuid>`` -> ``<uuid>``."""
    return work_uri.rstrip("/").rsplit("/", 1)[-1]


def _parse_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None
