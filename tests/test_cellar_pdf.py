"""PDF retrieval from CELLAR.

CELLAR is stubbed here with a mock transport -- these tests are about the manifestation
lookup, the item URL construction and the content guard, not about the live endpoint.
The live path is exercised by ``python -m euaia.ingest.pipeline --fetch-pdf``.
"""

from __future__ import annotations

import httpx
import pytest

from euaia.ingest.cellar import PDF_MANIFESTATION_TYPE, CellarClient, CellarError

MANIFESTATION = "http://publications.europa.eu/resource/cellar/abc-123.0001.02"
PDF_BYTES = b"%PDF-1.7\nfake pdf body\n%%EOF"


def _sparql_body(manifestations: list[str]) -> dict:
    return {"results": {"bindings": [{"manif": {"value": m}} for m in manifestations]}}


def _client(handler) -> CellarClient:
    return CellarClient(httpx.Client(transport=httpx.MockTransport(handler)))


class TestManifestationLookup:
    def test_returns_the_pdf_manifestation_uri(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert PDF_MANIFESTATION_TYPE in request.url.params["query"]
            return httpx.Response(200, json=_sparql_body([MANIFESTATION]))

        assert _client(handler).pdf_manifestation_uri("02024R1689-20260727") == MANIFESTATION

    def test_no_manifestation_is_an_error_not_an_empty_result(self):
        # Silently returning nothing would surface much later as a confusing download
        # failure; failing here names the actual problem.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_sparql_body([]))

        with pytest.raises(CellarError, match="no pdfa2a manifestation"):
            _client(handler).pdf_manifestation_uri("02024R1689-20260727")

    def test_the_language_is_part_of_the_query(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert "FRA" in request.url.params["query"]
            return httpx.Response(200, json=_sparql_body([MANIFESTATION]))

        _client(handler).pdf_manifestation_uri("02024R1689-20260727", language="FRA")


class TestFetchPdf:
    def test_downloads_the_item_below_the_manifestation(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if "sparql" in str(request.url):
                return httpx.Response(200, json=_sparql_body([MANIFESTATION]))
            return httpx.Response(200, content=PDF_BYTES)

        manifestation = _client(handler).fetch_pdf("02024R1689-20260727")

        assert manifestation.fmt == "pdf"
        assert manifestation.content == PDF_BYTES
        # The bytes live at <manifestation>/DOC_1, not at the manifestation itself.
        assert seen[-1].endswith(".0001.02/DOC_1")

    def test_the_item_url_is_upgraded_to_https(self):
        # The RDF graph returns http:// URIs; fetching those would redirect away from TLS.
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if "sparql" in str(request.url):
                return httpx.Response(200, json=_sparql_body([MANIFESTATION]))
            return httpx.Response(200, content=PDF_BYTES)

        _client(handler).fetch_pdf("02024R1689-20260727")
        assert seen[-1].startswith("https://")

    def test_non_pdf_content_is_rejected(self):
        # CELLAR answers the manifestation URI itself with RDF metadata; if the item path
        # ever changes shape we must fail loudly rather than cache an RDF file as a PDF.
        def handler(request: httpx.Request) -> httpx.Response:
            if "sparql" in str(request.url):
                return httpx.Response(200, json=_sparql_body([MANIFESTATION]))
            return httpx.Response(
                200, content=b"<rdf:RDF/>", headers={"content-type": "application/rdf+xml"}
            )

        with pytest.raises(CellarError, match="expected a PDF"):
            _client(handler).fetch_pdf("02024R1689-20260727")

    def test_the_hash_is_of_the_pdf_bytes(self):
        import hashlib

        def handler(request: httpx.Request) -> httpx.Response:
            if "sparql" in str(request.url):
                return httpx.Response(200, json=_sparql_body([MANIFESTATION]))
            return httpx.Response(200, content=PDF_BYTES)

        manifestation = _client(handler).fetch_pdf("02024R1689-20260727")
        assert manifestation.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
