from pathlib import Path

from pypdf import PdfReader


def load_policy_pages(path: Path) -> dict[int, str]:
    """Read the manual once, preserving its one-based PDF page numbers."""
    return {
        number: " ".join(page.extract_text().split())
        for number, page in enumerate(PdfReader(path).pages, start=1)
    }


def get_policy_page(pages: dict[int, str], number: int) -> dict[str, int | str]:
    """Return source evidence for a page selected by the agent."""
    return {"page": number, "text": pages.get(number, "")}
