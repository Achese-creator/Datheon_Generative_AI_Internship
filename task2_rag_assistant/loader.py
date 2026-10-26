from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from one document page or text file."""

    text: str
    source: str
    page_number: int | None = None


def load_document(file_path: str | Path) -> list[DocumentPage]:
    """Load a PDF or TXT file while preserving source metadata."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    pages: list[DocumentPage] = []

    if suffix == ".pdf":
        reader = PdfReader(str(path))

        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported.")

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                pages.append(
                    DocumentPage(
                        text=text,
                        source=path.name,
                        page_number=page_number,
                    )
                )

    elif suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            pages.append(
                DocumentPage(
                    text=text,
                    source=path.name,
                    page_number=None,
                )
            )

    else:
        raise ValueError(
            f"Unsupported file type: {suffix}. Use PDF or TXT."
        )

    if not pages:
        raise ValueError(
            f"No extractable text was found in {path.name}."
        )

    return pages
