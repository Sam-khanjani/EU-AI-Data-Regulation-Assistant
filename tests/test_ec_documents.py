"""The Commission document manifest and its fetch rules, offline: HTTP is replaced by a fake."""

from __future__ import annotations

import httpx
import pytest

from euaia.ingest import ec_documents as ec

PDF = b"%PDF-1.7\nbody"
PAGE_NOT_FOUND = b"<!DOCTYPE html><title>Page not found</title>"


class FakeClient:
    """Answers each URL from a mapping; anything unmapped raises, as httpx would."""

    def __init__(self, responses: dict[str, bytes | Exception]):
        self.responses, self.calls = responses, []

    def get(self, url: str):
        self.calls.append(url)
        answer = self.responses.get(url)
        if answer is None:
            raise httpx.ConnectError(f"unmapped {url}")
        if isinstance(answer, Exception):
            raise answer
        return httpx.Response(200, content=answer, request=httpx.Request("GET", url))


def doc(**kwargs) -> ec.ECDocument:
    return ec.ECDocument(
        filename=kwargs.pop("filename", "99_Test.pdf"),
        folder=kwargs.pop("folder", "Supporting_QA"),
        title="Test document",
        pdf_url=kwargs.pop("pdf_url", "https://example.invalid/doc.pdf"),
        landing_url="https://example.invalid/page",
        doc_date="2026-01-01",
        **kwargs,
    )


class TestManifest:
    def test_every_document_lands_somewhere_unique(self):
        paths = [str(d.path) for d in ec.DOCUMENTS]
        assert len(paths) == len(set(paths))

    def test_the_tree_matches_the_agreed_layout(self):
        by_folder: dict[str, list[str]] = {}
        for d in ec.DOCUMENTS:
            by_folder.setdefault(d.folder, []).append(d.filename)
        assert by_folder[""] == ["01_EU_AI_Act.pdf"]
        assert len(by_folder["Commission_Guidelines"]) == 5  # 03 is published as three parts
        assert len(by_folder["Codes_of_Practice"]) == 4
        assert len(by_folder["Supporting_QA"]) == 2

    def test_filenames_are_numbered_in_order(self):
        slots = [d.filename.split("_")[0] for d in ec.DOCUMENTS]
        assert slots == sorted(slots)

    def test_the_high_risk_guidelines_are_marked_draft(self):
        """They are out for consultation, and must never be presented as adopted law."""
        drafts = {d.filename for d in ec.DOCUMENTS if d.status == "draft"}
        assert drafts == {
            "03a_High_Risk_Guidelines_General.pdf",
            "03b_High_Risk_Guidelines_Annex_I.pdf",
            "03c_High_Risk_Guidelines_Annex_III.pdf",
        }

    def test_only_the_qa_pages_carry_an_html_fallback(self):
        """Everything else is published as PDF, so there is no HTML to fall back to."""
        assert {d.filename for d in ec.DOCUMENTS if d.html_url} == {
            "09_GPAI_QA.pdf",
            "10_Transparency_QA.pdf",
        }

    def test_the_act_itself_is_not_fetched_over_http(self):
        """It comes from the CELLAR cache the pipeline already maintains."""
        assert ec.AI_ACT.pdf_url is None


class TestFetchPrefersPdf:
    def test_pdf_is_used_when_it_works(self):
        d = doc(html_url="https://example.invalid/page.html")
        client = FakeClient({d.pdf_url: PDF})
        content, fmt, url = ec.fetch(d, client)
        assert (content, fmt, url) == (PDF, "pdf", d.pdf_url)
        assert client.calls == [d.pdf_url]  # the HTML was never requested

    def test_html_is_used_when_the_pdf_request_fails(self):
        d = doc(html_url="https://example.invalid/page.html")
        client = FakeClient({
            d.pdf_url: httpx.ConnectError("boom"),
            d.html_url: b"<html>answers</html>",
        })
        content, fmt, url = ec.fetch(d, client)
        assert (fmt, url) == ("html", d.html_url)
        assert content == b"<html>answers</html>"

    def test_a_pdf_url_answering_with_a_web_page_counts_as_a_failure(self):
        """The printable endpoint answers a bad node id with a 200 and an HTML error page.
        Saving that under a .pdf name would put a 'page not found' into the corpus."""
        d = doc(html_url="https://example.invalid/page.html")
        client = FakeClient({d.pdf_url: PAGE_NOT_FOUND, d.html_url: b"<html>real</html>"})
        content, fmt, _ = ec.fetch(d, client)
        assert (content, fmt) == (b"<html>real</html>", "html")

    def test_giving_up_names_the_document(self):
        d = doc(html_url=None)
        client = FakeClient({d.pdf_url: httpx.ConnectError("boom")})
        with pytest.raises(ec.FetchError, match="99_Test.pdf"):
            ec.fetch(d, client)


class TestDownload:
    def test_an_html_fallback_is_not_saved_under_a_pdf_name(self, tmp_path, monkeypatch):
        d = doc(filename="09_Fallback.pdf", html_url="https://example.invalid/page.html")
        client = FakeClient({d.pdf_url: PAGE_NOT_FOUND, d.html_url: b"<html>x</html>"})
        monkeypatch.setattr(ec, "DOCUMENTS", (d,))
        monkeypatch.setattr(ec.httpx, "Client", lambda **_kw: _nullcontext(client))

        manifest = ec.download(tmp_path)

        assert (tmp_path / "Supporting_QA" / "09_Fallback.html").exists()
        assert not (tmp_path / "Supporting_QA" / "09_Fallback.pdf").exists()
        assert "Supporting_QA/09_Fallback.html" in manifest["documents"]

    def test_existing_files_are_not_refetched(self, tmp_path, monkeypatch):
        d = doc(filename="02_Cached.pdf", folder="Commission_Guidelines")
        client = FakeClient({})  # any request would raise
        monkeypatch.setattr(ec, "DOCUMENTS", (d,))
        monkeypatch.setattr(ec.httpx, "Client", lambda **_kw: _nullcontext(client))
        target = tmp_path / "Commission_Guidelines" / "02_Cached.pdf"
        target.parent.mkdir(parents=True)
        target.write_bytes(PDF)

        manifest = ec.download(tmp_path)

        assert client.calls == []
        entry = manifest["documents"]["Commission_Guidelines/02_Cached.pdf"]
        assert entry["cached"] is True

    def test_the_manifest_records_provenance_for_each_file(self, tmp_path, monkeypatch):
        d = doc(filename="02_Doc.pdf", folder="Commission_Guidelines")
        monkeypatch.setattr(ec, "DOCUMENTS", (d,))
        monkeypatch.setattr(
            ec.httpx, "Client", lambda **_kw: _nullcontext(FakeClient({d.pdf_url: PDF}))
        )

        entry = ec.download(tmp_path)["documents"]["Commission_Guidelines/02_Doc.pdf"]

        assert entry["url"] == d.pdf_url
        assert entry["landing_url"] == d.landing_url
        assert entry["size"] == len(PDF)
        assert len(entry["sha256"]) == 64


class _nullcontext:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_exc):
        return False
