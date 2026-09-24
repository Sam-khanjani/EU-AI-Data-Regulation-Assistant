"""The European Commission's AI Act library: what it holds, and how to fetch it.

The AI Act itself comes from EUR-Lex through :mod:`euaia.ingest.cellar`, which resolves a
version over SPARQL and pulls it from CELLAR. The Commission's own material -- guidelines,
codes of practice, Q&A -- has no such machinery. It is published as ordinary files behind the
newsroom redirector, so it is listed here by hand and fetched directly.

The manifest below is the authoritative list. ``data/raw`` is not tracked in git because it is
reproducible, which only holds while something records where each file came from: this module
is that record.

PDF is preferred over HTML wherever both exist, so the whole corpus stays on one reader. That
rule costs nothing for the guidelines and codes, which are published as PDF anyway. It matters
for the Q&A pages, which are HTML articles -- but every one exposes a Drupal node id, and
``/en/node/<id>/printable/pdf`` renders it server-side. :data:`ECDocument.html_url` is the
fallback for when that endpoint is missing or breaks, and the fetch records which route it
actually took.

This is the corpus' one download step: it also fetches the Act's own PDFs from EUR-Lex into
the pipeline's cache. Run it first, then ingest once:

    uv run python -m euaia.ingest.ec_documents
    uv run python -m euaia.ingest.pipeline

What these documents are, before anyone indexes them
----------------------------------------------------

Measured on the files this module fetches. Each of these breaks an assumption the AI Act path
is allowed to make, so they are recorded here rather than rediscovered later.

**Three of them are drafts.** The high-risk classification guidelines were published for
consultation on 19 May 2026, with the final text expected at the end of 2026; two still carry
an unfilled ``Brussels, XXX`` where the adoption date goes. :attr:`ECDocument.status` marks
them. Indexing a draft beside the Act without carrying that status through to the answer would
put an unadopted document behind the same verified-citation badge as binding law, which is
precisely the claim this project exists to make carefully.

**Guidelines and codes are not structured like a regulation.** The Act divides into chapters,
articles, paragraphs and annexes, and ``structural_unit.unit_path`` encodes that as
``CH_III/ART_6/PAR_2``. These documents divide into decimal-numbered sections (``2.3.1.``),
and the codes into Commitments and Measures. Neither the ``unit_type`` enum nor the path
convention covers them, so a citation target for this material has to be designed, not
inherited.

**One has no bookmark outline.** Eleven of the twelve carry publisher-authored outlines, which
is what :mod:`euaia.ingest.pdf` reads instead of guessing headings from fonts. The Article 50
transparency guidelines (51 pages) carry none. They do number their headings consistently, so
a numbering-run reader works -- the approach the recitals already use, where the parse is only
accepted if the sequence is complete -- but it is a second reader, not the existing one.

**There are no per-provision deep links.** Every AI Act unit carries an EUR-Lex anchor that
was verified to resolve. Nothing here has an equivalent: the best available citation target is
the landing page plus a page number, which is what ``structural_unit.page`` holds.

**Change detection cannot use the CELLAR route.** :mod:`euaia.ingest.cellar` resolves new
versions over SPARQL, which only works for documents with a CELEX. The Commission revises
these in place under a stable URL -- the GPAI guidelines page is dated July 2025 and serves a
PDF revised in November 2025, with no version label anywhere. Comparing the SHA-256 recorded
in the manifest is the only signal available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx

from euaia.config import settings

log = logging.getLogger(__name__)

Folder = Literal["", "Commission_Guidelines", "Codes_of_Practice", "Supporting_QA"]
Status = Literal["final", "draft"]

Authority = Literal["law", "guidance", "code"]
"""How much weight a document's text carries, which is not the same as how relevant it is.

``law``
    The Regulation itself. States what is legally required.
``guidance``
    The Commission's own interpretation -- guidelines and Q&A. Authoritative about how the
    Commission reads the Act, but not binding in itself.
``code``
    Codes of practice. Voluntary, and adherence is evidence of compliance rather than
    compliance itself.

Answering "what is required?" mainly from a voluntary code would be wrong however well that
code matched the question, so this ordering is carried through retrieval into the answer.
The Q&A are ``guidance`` rather than a tier of their own: they are Commission-published
interpretation, just informally written.
"""

_NEWSROOM = "https://ec.europa.eu/newsroom/dae/redirection/document"
_DIGITAL_STRATEGY = "https://digital-strategy.ec.europa.eu/en"

DOWNLOAD_ROOT = "AI_Act"
"""Subdirectory of ``raw_data_dir`` holding the tree. Named here because
:mod:`euaia.ingest.sources` addresses these files by path."""

MANIFEST_NAME = "manifest.json"
"""Written into the download root: what was fetched, from where, and with what hash."""


@dataclass(frozen=True, slots=True)
class ECDocument:
    """One publication in the Commission's AI Act library."""

    filename: str
    folder: Folder
    title: str
    pdf_url: str | None
    landing_url: str
    doc_date: str
    """Date on the document itself, which is not always its page's publication date -- the
    GPAI guidelines page is dated July 2025 and carries a PDF revised in November."""

    authority: Authority = "guidance"
    status: Status = "final"
    html_url: str | None = None
    """Used only when :attr:`pdf_url` is absent or fails; see the module docstring."""

    notes: str = ""

    @property
    def path(self) -> Path:
        return Path(self.folder) / self.filename if self.folder else Path(self.filename)

    @property
    def source_key(self) -> str:
        """``03a_High_Risk_Guidelines_General.pdf`` -> ``ec-high-risk-guidelines-general``.

        Derived rather than written out so a file and its source row can never drift apart.
        """
        stem = Path(self.filename).stem.split("_", 1)[1]
        return "ec-" + stem.lower().replace("_", "-")

    @property
    def short_title(self) -> str:
        """``05_GPAI_Code_Transparency.pdf`` -> ``GPAI Code Transparency``.

        The full title is a sentence ("Guidelines on the scope of the obligations for..."),
        which is unreadable inside a citation link and wasteful in every chunk breadcrumb.
        """
        return Path(self.filename).stem.split("_", 1)[1].replace("_", " ")

    @property
    def version_label(self) -> str:
        """Shown in the evidence block the answer model reads, so a draft announces itself."""
        return f"{'draft' if self.status == 'draft' else 'published'} {self.doc_date}"


# --------------------------------------------------------------------------- the manifest

AI_ACT = ECDocument(
    filename="01_EU_AI_Act.pdf",
    folder="",
    title="Regulation (EU) 2024/1689 (Artificial Intelligence Act), consolidated",
    pdf_url=None,  # fetched through CELLAR, not the newsroom; see _copy_ai_act
    landing_url="https://eur-lex.europa.eu/legal-content/EN/ALL/?uri=CELEX:32024R1689",
    doc_date="2026-07-27",
    authority="law",
    notes="Fetched from EUR-Lex into the pipeline's cache, then copied here.",
)

GUIDELINES = (
    ECDocument(
        filename="02_GPAI_Guidelines.pdf",
        folder="Commission_Guidelines",
        title="Guidelines on the scope of the obligations for general-purpose AI models",
        pdf_url=f"{_NEWSROOM}/118340",
        landing_url=(
            f"{_DIGITAL_STRATEGY}/library/"
            "guidelines-scope-obligations-providers-general-purpose-ai-models-under-ai-act"
        ),
        doc_date="2025-11-19",
    ),
    # Published as three separate documents, so they stay three files. The Annex III part
    # alone is 148 pages -- larger than the Act's operative text.
    ECDocument(
        filename="03a_High_Risk_Guidelines_General.pdf",
        folder="Commission_Guidelines",
        title="Draft guidelines on the classification of high-risk AI systems: general principles",
        pdf_url=f"{_NEWSROOM}/128559",
        landing_url=(
            f"{_DIGITAL_STRATEGY}/library/"
            "draft-commission-guidelines-classification-high-risk-ai-systems"
        ),
        doc_date="2026-05-19",
        status="draft",
        notes="Out for targeted consultation; final text expected end of 2026.",
    ),
    ECDocument(
        filename="03b_High_Risk_Guidelines_Annex_I.pdf",
        folder="Commission_Guidelines",
        title="Draft guidelines on the classification of high-risk AI systems: Annex I",
        pdf_url=f"{_NEWSROOM}/128560",
        landing_url=(
            f"{_DIGITAL_STRATEGY}/library/"
            "draft-commission-guidelines-classification-high-risk-ai-systems"
        ),
        doc_date="2026-05-19",
        status="draft",
        notes="Still carries an unfilled 'Brussels, XXX' adoption date.",
    ),
    ECDocument(
        filename="03c_High_Risk_Guidelines_Annex_III.pdf",
        folder="Commission_Guidelines",
        title="Draft guidelines on the classification of high-risk AI systems: Annex III",
        pdf_url=f"{_NEWSROOM}/128561",
        landing_url=(
            f"{_DIGITAL_STRATEGY}/library/"
            "draft-commission-guidelines-classification-high-risk-ai-systems"
        ),
        doc_date="2026-05-19",
        status="draft",
        notes="Still carries an unfilled 'Brussels, XXX' adoption date.",
    ),
    ECDocument(
        filename="04_Transparency_Guidelines.pdf",
        folder="Commission_Guidelines",
        title=(
            "Guidelines on the implementation of the transparency obligations for certain "
            "AI systems under Article 50 of the AI Act"
        ),
        pdf_url=f"{_NEWSROOM}/131215",
        landing_url=(
            f"{_DIGITAL_STRATEGY}/library/"
            "guidelines-transparency-obligations-providers-and-deployers-ai-systems"
        ),
        doc_date="2026-07-20",
        notes="The only document here published without a bookmark outline.",
    ),
)

CODES = (
    ECDocument(
        filename="05_GPAI_Code_Transparency.pdf",
        folder="Codes_of_Practice",
        title="General-Purpose AI Code of Practice: Transparency chapter",
        pdf_url=f"{_NEWSROOM}/118120",
        landing_url=f"{_DIGITAL_STRATEGY}/policies/contents-code-gpai",
        doc_date="2025-07-10",
        authority="code",
    ),
    ECDocument(
        filename="06_GPAI_Code_Copyright.pdf",
        folder="Codes_of_Practice",
        title="General-Purpose AI Code of Practice: Copyright chapter",
        pdf_url=f"{_NEWSROOM}/118115",
        landing_url=f"{_DIGITAL_STRATEGY}/policies/contents-code-gpai",
        doc_date="2025-07-10",
        authority="code",
    ),
    ECDocument(
        filename="07_GPAI_Code_Safety.pdf",
        folder="Codes_of_Practice",
        title="General-Purpose AI Code of Practice: Safety and Security chapter",
        pdf_url=f"{_NEWSROOM}/118119",
        landing_url=f"{_DIGITAL_STRATEGY}/policies/contents-code-gpai",
        doc_date="2025-07-10",
        authority="code",
    ),
    ECDocument(
        filename="08_Transparency_Code_AI_Content.pdf",
        folder="Codes_of_Practice",
        title="Code of Practice on Transparency of AI-Generated Content",
        pdf_url=f"{_NEWSROOM}/129555",
        landing_url=f"{_DIGITAL_STRATEGY}/policies/code-practice-ai-generated-content",
        doc_date="2026-06-10",
        authority="code",
    ),
)

QA = (
    ECDocument(
        filename="09_GPAI_QA.pdf",
        folder="Supporting_QA",
        title="General-Purpose AI Models in the AI Act - Questions & Answers",
        pdf_url=f"{_DIGITAL_STRATEGY}/node/13148/printable/pdf",
        html_url=f"{_DIGITAL_STRATEGY}/faqs/general-purpose-ai-models-ai-act-questions-answers",
        landing_url=f"{_DIGITAL_STRATEGY}/faqs/general-purpose-ai-models-ai-act-questions-answers",
        doc_date="2025-09-09",
        notes="HTML article; PDF is the site's own server-side rendering of it.",
    ),
    ECDocument(
        filename="10_Transparency_QA.pdf",
        folder="Supporting_QA",
        title="Transparency obligations under Article 50 of the AI Act - Questions & Answers",
        pdf_url=f"{_DIGITAL_STRATEGY}/node/17084/printable/pdf",
        html_url=f"{_DIGITAL_STRATEGY}/faqs/transparency-obligations-under-article-50-ai-act",
        landing_url=f"{_DIGITAL_STRATEGY}/faqs/transparency-obligations-under-article-50-ai-act",
        doc_date="2026-08-06",
        notes="HTML article; PDF is the site's own server-side rendering of it.",
    ),
)

DOCUMENTS: tuple[ECDocument, ...] = (AI_ACT, *GUIDELINES, *CODES, *QA)


# --------------------------------------------------------------------------- fetching


class FetchError(RuntimeError):
    """A document could not be retrieved in any format."""


def _is_pdf(content: bytes) -> bool:
    """The newsroom redirector serves PDFs with no ``Content-Type``, and the Drupal printable
    endpoint answers a bad node id with an HTML 'page not found' under a 200. Neither status
    codes nor headers can be trusted, so the bytes are checked instead."""
    return content.startswith(b"%PDF-")


def _get(client: httpx.Client, url: str) -> bytes:
    response = client.get(url)
    response.raise_for_status()
    return response.content


def fetch(doc: ECDocument, client: httpx.Client) -> tuple[bytes, str, str]:
    """Retrieve one document. Returns its bytes, the format used, and the URL it came from.

    PDF first, HTML only if that fails -- and a PDF request that quietly returns a web page
    counts as a failure, not as HTML.
    """
    if doc.pdf_url:
        try:
            content = _get(client, doc.pdf_url)
            if _is_pdf(content):
                return content, "pdf", doc.pdf_url
            log.warning("%s: %s did not return a PDF; falling back", doc.filename, doc.pdf_url)
        except httpx.HTTPError as exc:
            log.warning("%s: PDF fetch failed (%s); falling back", doc.filename, exc)

    if doc.html_url:
        return _get(client, doc.html_url), "html", doc.html_url

    raise FetchError(f"{doc.filename}: no usable source (tried {doc.pdf_url or 'nothing'})")


def _fetch_eurlex_sources() -> None:
    """Download the Act's EUR-Lex PDFs into the pipeline's cache, if not already there.

    This makes this module the one download step for the whole corpus, so ingestion runs once,
    after it, with everything already on disk. The files go exactly where the pipeline's own
    fetch would put them (by CELEX under ``raw_data_dir``), so ingestion finds them cached and
    downloads nothing. Needs no database.
    """
    # Imported here: `sources` imports this module, so a top-level import would be circular.
    from euaia.ingest import pipeline, sources
    from euaia.ingest.cellar import CellarClient

    with CellarClient() as client:
        for spec in sources.ALL_SOURCES:
            if not spec.local_file:
                pipeline.fetch_pdf_content(client, pipeline.resolve_target(client, spec))


def _copy_ai_act(destination: Path) -> dict[str, object] | None:
    """Place the Act beside the Commission material, from the pipeline's cache.

    The pipeline caches it under ``raw_data_dir`` by CELEX, and addresses it by that name, so
    the cache is left exactly where it is and this is a copy rather than a move.
    """
    _fetch_eurlex_sources()
    candidates = sorted(settings.raw_data_dir.glob("02024R1689-*.ENG.pdf"), reverse=True)
    if not candidates:
        log.warning("%s: no consolidated act in %s after fetching from EUR-Lex",
                    AI_ACT.filename, settings.raw_data_dir)
        return None

    source = candidates[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    content = destination.read_bytes()
    log.info("Copied %s -> %s (%d bytes)", source.name, AI_ACT.filename, len(content))
    return {"fmt": "pdf", "url": f"cache:{source.name}", "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest()}


def download(root: Path, *, refresh: bool = False) -> dict[str, object]:
    """Download every document into ``root``, and return the manifest describing the result."""
    entries: dict[str, object] = {}
    headers = {"User-Agent": settings.cellar_user_agent}

    with httpx.Client(follow_redirects=True, timeout=settings.cellar_timeout_seconds,
                      headers=headers) as client:
        for doc in DOCUMENTS:
            destination = root / doc.path
            record: dict[str, object] = {
                "title": doc.title, "status": doc.status, "doc_date": doc.doc_date,
                "landing_url": doc.landing_url, "notes": doc.notes,
            }

            if destination.exists() and not refresh:
                content = destination.read_bytes()
                log.info("Have %s (%d bytes)", doc.path, len(content))
                record |= {"fmt": destination.suffix.lstrip("."), "cached": True,
                           "size": len(content),
                           "sha256": hashlib.sha256(content).hexdigest()}
                entries[str(doc.path).replace("\\", "/")] = record
                continue

            if doc is AI_ACT:
                if copied := _copy_ai_act(destination):
                    entries[str(doc.path).replace("\\", "/")] = record | copied
                continue

            content, fmt, url = fetch(doc, client)
            # An HTML fallback must not be saved under a .pdf name.
            if fmt == "html":
                destination = destination.with_suffix(".html")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            log.info("Fetched %s as %s (%d bytes)", destination.name, fmt, len(content))
            record |= {"fmt": fmt, "url": url, "size": len(content),
                       "sha256": hashlib.sha256(content).hexdigest()}
            entries[str(destination.relative_to(root)).replace("\\", "/")] = record

    return {"retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"), "documents": entries}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=settings.raw_data_dir / DOWNLOAD_ROOT,
                        help="where the tree is written (default: data/raw/AI_Act)")
    parser.add_argument("--refresh", action="store_true",
                        help="re-download files that are already present")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.root.mkdir(parents=True, exist_ok=True)

    manifest = download(args.root, refresh=args.refresh)
    (args.root / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    documents = manifest["documents"]
    assert isinstance(documents, dict)
    total = sum(int(entry["size"]) for entry in documents.values())
    print(f"\n{len(documents)} documents, {total / 1_048_576:.1f} MB, in {args.root}")
    print(f"manifest: {args.root / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
