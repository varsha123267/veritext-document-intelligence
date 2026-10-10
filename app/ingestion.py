"""
Document ingestion: parsing raw uploaded bytes into clean text, and
splitting that text into overlapping chunks suitable for embedding
and retrieval.
"""
import io
import re
from typing import List, Optional, Tuple

from pypdf import PdfReader


def parse_file_pages(filename: str, raw_bytes: bytes) -> List[Tuple[Optional[int], str]]:
    """Extract text from an uploaded file's raw bytes, split by page where
    that concept applies.

    Returns a list of (page_number, text) tuples. For PDFs, page_number is
    1-indexed and there's one tuple per page. For .txt/.md (which have no
    pages), page_number is None and there's a single tuple for the whole
    file. Downstream chunking preserves this page number on every chunk it
    produces, which is what lets citations say "Page 3" instead of just a
    vague "source 1".
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages: List[Tuple[Optional[int], str]] = []
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append((i + 1, text))
        if not pages:
            raise ValueError(
                "No extractable text found in this PDF (it may be a scanned "
                "image without an OCR text layer)."
            )
        return pages

    if lower.endswith(".txt") or lower.endswith(".md"):
        text = raw_bytes.decode("utf-8", errors="ignore")
        return [(None, text)]

    raise ValueError(f"Unsupported file type for '{filename}'. Use .txt, .md, or .pdf.")


def parse_file(filename: str, raw_bytes: bytes) -> str:
    """Convenience wrapper: all the text from a file as one plain string,
    for callers (like the document summary) that don't care about pages."""
    pages = parse_file_pages(filename, raw_bytes)
    return "\n\n".join(text for _, text in pages)


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, target_len: int = 800, overlap: int = 120) -> List[str]:
    """
    Paragraph-aware chunking: greedily pack paragraphs into chunks close to
    `target_len` characters, splitting overlong paragraphs by sentence, with
    a small character overlap carried between consecutive chunks so context
    isn't lost at chunk boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if len(buf.strip()) > 20:
            chunks.append(buf.strip())
        buf = ""

    for para in paragraphs:
        if len(para) > target_len * 1.6:
            sentences = re.split(r"(?<=[.?!])\s+", para)
            sbuf = buf
            for sentence in sentences:
                if len(sbuf) + len(sentence) + 1 > target_len:
                    if len(sbuf.strip()) > 20:
                        chunks.append(sbuf.strip())
                    sbuf = sbuf[-overlap:] + " " + sentence
                else:
                    sbuf += " " + sentence
            buf = sbuf
            continue

        if len(buf) + len(para) + 2 > target_len:
            flush()
            buf = para
        else:
            buf = f"{buf}\n\n{para}" if buf else para

    flush()
    return chunks if chunks else [text.strip()]


def chunk_pages(
    pages: List[Tuple[Optional[int], str]], target_len: int = 800, overlap: int = 120
) -> List[Tuple[Optional[int], str]]:
    """Chunk each page's text separately (via chunk_text) so every resulting
    chunk can carry its own page number. For a document with no page concept
    (page_number is None throughout), this behaves the same as chunking the
    whole document at once."""
    result: List[Tuple[Optional[int], str]] = []
    for page_number, text in pages:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        for chunk in chunk_text(cleaned, target_len=target_len, overlap=overlap):
            result.append((page_number, chunk))
    return result


def extract_keywords(chunks: List[str], top_n: int = 5) -> List[str]:
    """Cheap keyword extraction: most frequent non-stopword tokens across a
    document's chunks. This is a lightweight complement to the vector store's
    embeddings, used only for the document index-card UI, not for retrieval."""
    stop = STOPWORDS
    freq = {}
    for chunk in chunks:
        for tok in re.findall(r"[a-zA-Z0-9]+", chunk.lower()):
            if len(tok) > 2 and tok not in stop:
                freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]


STOPWORDS = set(
    "the and for are but not you all can had her was one our out day get has him his how man new now old see two "
    "way who boy did its let put say she too use with this that these those from into over under again further "
    "once here there when where why what which while about above below between during before after such same "
    "other some more most only own than too very just also does did doing have has had having been being do is "
    "are was were will would could should".split()
)
