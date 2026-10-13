"""FinanceBench loading, PDF parsing, page-aware chunking, and entity (company / year) detection."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

# Hand-written entity dictionary for the 40 companies in the corpus: the official name, common short names
# and the stock ticker. Matched case-sensitively on word boundaries (tickers like BA or KO would otherwise
# hit ordinary words).
COMPANY_ALIASES = {
    "3M": ["3M", "MMM"],
    "AES Corporation": ["AES"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "Activision Blizzard": ["Activision Blizzard", "Activision", "ATVI"],
    "Adobe": ["Adobe", "ADBE"],
    "Amazon": ["Amazon", "AMZN"],
    "Amcor": ["Amcor", "AMCR"],
    "American Express": ["American Express", "AmEx", "AMEX", "AXP"],
    "American Water Works": ["American Water Works", "American Water", "AWK"],
    "Apple": ["Apple", "AAPL"],
    "Best Buy": ["Best Buy", "BBY"],
    "Block": ["Block", "Square", "SQ"],
    "Boeing": ["Boeing", "BA"],
    "CVS Health": ["CVS Health", "CVS"],
    "Coca-Cola": ["Coca-Cola", "Coca Cola", "Coke", "KO"],
    "Corning": ["Corning", "GLW"],
    "Costco": ["Costco", "COST"],
    "FedEx": ["FedEx", "FDX"],
    "Foot Locker": ["Foot Locker", "FL"],
    "General Mills": ["General Mills", "GIS"],
    "Intel": ["Intel", "INTC"],
    "JPMorgan": ["JPMorgan", "JP Morgan", "JPM"],
    "Johnson & Johnson": ["Johnson & Johnson", "Johnson and Johnson", "J&J", "JnJ", "JNJ"],
    "Kraft Heinz": ["Kraft Heinz", "Kraft", "KHC"],
    "Lockheed Martin": ["Lockheed Martin", "Lockheed", "LMT"],
    "MGM Resorts": ["MGM Resorts", "MGM"],
    "McDonalds": ["McDonald's", "McDonalds", "MCD"],
    "Microsoft": ["Microsoft", "MSFT"],
    "Netflix": ["Netflix", "NFLX"],
    "Nike": ["Nike", "NKE"],
    "Oracle": ["Oracle", "ORCL"],
    "PG&E Corporation": ["PG&E", "PCG"],
    "Paypal": ["PayPal", "Paypal", "PYPL"],
    "PepsiCo": ["PepsiCo", "Pepsi", "PEP"],
    "Pfizer": ["Pfizer", "PFE"],
    "Salesforce": ["Salesforce", "CRM"],
    "Ulta Beauty": ["Ulta Beauty", "Ulta", "ULTA"],
    "Verizon": ["Verizon", "VZ"],
    "Walmart": ["Walmart", "WMT"],
    "eBay": ["eBay", "EBAY"],
}
_ALIAS_RE = [(company, re.compile(r"(?<![\w&])" + re.escape(a) + r"(?![\w&])"))
             for company, aliases in COMPANY_ALIASES.items() for a in aliases]
_ALIASES_OF = {c: [rx for company, rx in _ALIAS_RE if company == c] for c in COMPANY_ALIASES}

# SEC filers that are not in the corpus: questions about them cannot be answered from it.
OUTSIDE_COMPANIES = ["Tesla", "Nvidia", "Starbucks", "Disney", "Ford", "Uber", "Airbnb", "Qualcomm",
                     "Caterpillar", "Visa"]
_YEAR_RE = re.compile(r"(?<!\d)(20[0-3]\d)(?!\d)")


def load_questions(fb: Path) -> list[dict]:
    with open(fb / "data" / "financebench_open_source.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_docinfo(fb: Path) -> dict[str, dict]:
    with open(fb / "data" / "financebench_document_information.jsonl") as f:
        return {r["doc_name"]: r for r in (json.loads(line) for line in f if line.strip())}


def doc_meta(doc_name: str, info: dict[str, dict]) -> dict:
    """company / doc_type / period for one PDF. A few PDFs are missing from the info file; they follow the
    same {COMPANY}_{PERIOD}_{TYPE} naming, so the company comes from another document with that prefix."""
    if doc_name in info:
        r = info[doc_name]
        return {"company": r["company"], "doc_type": str(r["doc_type"]).upper(), "period": str(r["doc_period"])}
    parts = doc_name.split("_")
    prefix = parts[0]
    period = parts[1] if len(parts) > 1 else ""
    dtype = parts[2] if len(parts) > 2 else ""
    company = next((r["company"] for name, r in info.items() if name.split("_")[0] == prefix), prefix)
    return {"company": company, "doc_type": dtype.upper(), "period": period}


def clean_page(text: str) -> str:
    """Collapse runs of spaces and drop empty lines; keep line breaks, which carry table rows."""
    lines = (re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def parse_pdf(path: str) -> list[str]:
    """One cleaned string per page, in page order (index = zero-based page number, as in FinanceBench)."""
    import pymupdf
    with pymupdf.open(path) as doc:
        return [clean_page(page.get_text("text", sort=True)) for page in doc]


def page_lines(text: str, max_words: int = 120) -> list[str]:
    """Non-empty lines of a page; a line longer than `max_words` words is cut into word windows."""
    lines: list[str] = []
    for line in text.splitlines():
        words = line.split()
        if len(words) <= max_words:
            if words:
                lines.append(line)
        else:
            lines.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    return lines


def chunk_lines(lines: list[str], sizes: list[int], max_size: int, overlap: int) -> list[str]:
    """Group consecutive lines into chunks of at most `max_size` (a single larger line is its own chunk);
    each chunk after the first repeats up to `overlap` worth of the previous chunk's trailing lines.
    Callers pass one page at a time, so a chunk never crosses a page and has exactly one page number."""
    assert 0 <= overlap < max_size and len(lines) == len(sizes)
    chunks: list[str] = []
    cur: list[int] = []
    cur_size = 0
    for i, n in enumerate(sizes):
        if cur and cur_size + n > max_size:
            chunks.append("\n".join(lines[j] for j in cur))
            tail: list[int] = []
            tail_size = 0
            for j in reversed(cur):
                if tail_size + sizes[j] > overlap:
                    break
                tail.insert(0, j)
                tail_size += sizes[j]
            cur, cur_size = tail, tail_size
        cur.append(i)
        cur_size += n
    if cur:
        chunks.append("\n".join(lines[j] for j in cur))
    return chunks


def chunk_page(text: str, max_size: int = 300, overlap: int = 50, sizes_of=None) -> list[str]:
    """Chunks of one page. Size is counted in words unless `sizes_of(lines) -> list[int]` is given
    (prep passes the embedding tokenizer, since digits tokenize one by one and tables run long)."""
    lines = page_lines(text)
    sizes = sizes_of(lines) if sizes_of else [len(line.split()) for line in lines]
    return chunk_lines(lines, sizes, max_size, overlap)


def chunk_header(meta: dict, page: int) -> str:
    """Prepended to each chunk for indexing, so a page that never names its company or year still matches."""
    return f"{meta['company']} {meta['doc_type']} {meta['period']}, page {page + 1}"


def detect_companies(question: str) -> list[str]:
    found = []
    for company, rx in _ALIAS_RE:
        if company not in found and rx.search(question):
            found.append(company)
    return found


def detect_years(question: str) -> list[str]:
    return sorted(set(_YEAR_RE.findall(question.replace("FY", " "))))


def _words(s: str) -> list[str]:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def evidence_overlap(evidence: str, page_text: str) -> float:
    """Share of the evidence's words that occur on the page (bag of words, so line breaks don't matter)."""
    ev = _words(evidence)
    if not ev:
        return 0.0
    page = set(_words(page_text))
    return sum(w in page for w in ev) / len(ev)


def make_unanswerable(questions: list[dict], n_each: int = 25, seed: int = 0, shift: int = 10) -> list[dict]:
    """Questions the corpus cannot answer, made from FinanceBench questions: every year moved `shift` years
    later (no filing is that recent), or the company replaced by one outside the corpus. The two kinds use
    different source questions."""
    order = list(questions)
    random.Random(seed).shuffle(order)
    out: list[dict] = []
    year_src = [q for q in order if _YEAR_RE.search(q["question"])][:n_each]
    for q in year_src:
        text = _YEAR_RE.sub(lambda m: str(int(m.group(1)) + shift), q["question"])
        out.append({"financebench_id": "unans-year-" + q["financebench_id"].split("_")[-1], "question": text,
                    "kind": "year", "source": q["financebench_id"]})
    used = {q["financebench_id"] for q in year_src}
    co_src = [q for q in order if q["financebench_id"] not in used and detect_companies(q["question"])][:n_each]
    for k, q in enumerate(co_src):
        name, text = OUTSIDE_COMPANIES[k % len(OUTSIDE_COMPANIES)], q["question"]
        for company in detect_companies(text):
            for rx in _ALIASES_OF[company]:
                text = rx.sub(name, text)
        out.append({"financebench_id": "unans-company-" + q["financebench_id"].split("_")[-1], "question": text,
                    "kind": "company", "source": q["financebench_id"]})
    return out
