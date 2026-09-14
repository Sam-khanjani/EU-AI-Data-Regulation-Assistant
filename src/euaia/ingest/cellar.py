"""Client for the EU Publications Office CELLAR repository.

Two capabilities:

* **Version resolution** via the SPARQL endpoint -- find every consolidated version of a
  regulation and pick the newest. This is the change-detection signal: consolidated acts
  are separate works with their own CELEX, so "is there a new version?" is a metadata
  question, not a diff of a rendered page.
* **Content retrieval** via the REST API. We fetch the PDF rendering (structured
  article-level markup) or PDF/A.

**Why retrieving a PDF takes two requests.** CELLAR models a document as a
*work* with one *expression* per language, and one *manifestation* per format. Requesting
the work URI with ``Accept: application/pdf`` returns 404 -- the PDF is not negotiable at
that level. It has to be addressed as its own manifestation, and the file itself lives one
level below that again, at ``<manifestation>/DOC_1``. So PDF retrieval is: SPARQL for the
manifestation whose ``cdm:manifestation_type`` is ``pdfa2a``, then fetch its item.

The manifestation suffixes (``.0001.01``, ``.0001.02``, ...) are *not* stable across
documents -- PDF happened to be ``.02`` for the AI Act, but that ordering
is an accident of registration order. Always resolve by manifestation type, never by suffix.

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
import logging
from dataclasses import dataclass

import httpx
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from euaia.config import settings

log = logging.getLogger(__name__)

CDM = "http://publications.europa.eu/ontology/cdm#"

# cdm:manifestation_type of the PDF rendering. PDF/A-2a is the archival profile the
# Publications Office registers for consolidated acts.
PDF_MANIFESTATION_TYPE = "pdfa2a"


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
    fmt: str  # currently always "pdf"
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

    Requests are issued sequentially, which keeps us inside CELLAR's guidance of fewer than
    five concurrent requests by construction.
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
        retry=retry_if_exception(_is_retryable),
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

    def pdf_manifestation_uri(self, celex: str, language: str = "ENG") -> str:
        """Find the URI of the PDF manifestation for one version's CELEX.

        Resolved by manifestation *type* rather than by URI suffix -- see the module
        docstring for why the suffix cannot be relied on.
        """
        rows = self.sparql(
            f"""
            PREFIX cdm: <{CDM}>
            SELECT DISTINCT ?manif WHERE {{
              ?work cdm:resource_legal_id_celex ?celex .
              FILTER(STR(?celex) = "{celex}")
              ?expr cdm:expression_belongs_to_work ?work ;
                    cdm:expression_uses_language ?lang .
              FILTER(CONTAINS(STR(?lang), "{language}"))
              ?manif cdm:manifestation_manifests_expression ?expr ;
                     cdm:manifestation_type ?type .
              FILTER(STR(?type) = "{PDF_MANIFESTATION_TYPE}")
            }}
            LIMIT 1
            """
        )
        if not rows:
            raise CellarError(
                f"no {PDF_MANIFESTATION_TYPE} manifestation found for CELEX {celex} "
                f"in language {language}"
            )
        return rows[0]["manif"]

    def fetch_pdf(self, celex: str, language: str = "ENG") -> Manifestation:
        """Download the PDF/A rendering of one version, addressed by its CELEX.

        Takes a CELEX rather than a cellar id because the PDF is found through the
        expression/manifestation graph, which is keyed on the work's CELEX -- the cellar id
        of the work is not part of that lookup.
        """
        manifestation = self.pdf_manifestation_uri(celex, language)
        # The manifestation URI is the *description* of the file; the bytes live one level
        # below it. https rather than the http the graph returns, so the fetch is not a
        # redirect away from TLS.
        item = f"{manifestation.replace('http://', 'https://', 1)}/DOC_1"
        resp = self._get_item(item)
        content = resp.content
        if not content.startswith(b"%PDF-"):
            raise CellarError(
                f"expected a PDF from {item}, got {resp.headers.get('content-type')!r}"
            )
        return Manifestation(
            content=content,
            fmt="pdf",
            sha256=hashlib.sha256(content).hexdigest(),
            filename=f"{celex}.{language}.pdf",
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _get_item(self, url: str) -> httpx.Response:
        """GET an absolute CELLAR item URL (as opposed to a work id under the base)."""
        resp = self._client.get(url)
        if not _ok(resp):
            resp.raise_for_status()
        return resp


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
