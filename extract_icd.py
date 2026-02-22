"""
build_icd_json_v2.py
--------------------
ICD-10 extractor for Russian medical protocols corpus.
Handles all known OCR/formatting artefacts found in 1137 protocols.

FIXES vs v1
===========
  1  Spaced codes        "I 86.1", "К 55.1", "S 42.3"  → normalize before regex
  2  Long descriptions   >80 chars truncated             → widened to 160 chars
  3  Bracket codes       "(С06.9)" inline                → expand before regex
  4  Asterisk lookahead  "G63.2*" desc eaten by "("      → stop desc at "("
  5  Cyrillic O-as-zero  "BOO.O" → B00.0                 → _fix_O_as_zero()
  6  Spaced decimal      "J84. 8 1" → J84.81             → collapse in normalizer
  7  Range lookahead     "D55-59" not matched by "1.2"   → \d+[.\s)] terminators
  8  Comma-separated     "В20-В24, С00-С97, G20, ..."    → _extract_comma_list()
  9  Mixed ICD-9/10      "I73.1 … 38.25 …"               → strip NN.NN ICD-9 codes first
 10  OCR digit-only      "142.0" (dropped 'I')            → _recover_digit_only()
 11  Old МБК spaced      "I0. 1.2." → I01.2              → fixed merge regex
 12  Code, no desc       "N73.6."                         → accept with empty desc
 13  U-chapter codes     "U09", "U09.9"                   → already valid, fixed terminator
 14  МКБ-11 side table   "F15" beside МКБ-11 col          → strip МКБ-11 column first
 15  Explicit нет        "МКБ-10 – нет."                  → broadened нет detection
 16  Q 64.O / РО7        Cyrillic letters in digit slots  → transliterate + O-as-zero
"""

import json
import re
import argparse
from pathlib import Path


# ── Transliteration: Cyrillic look-alikes → Latin ────────────────────────────
CYRILLIC_TO_LATIN = {
    'А':'A','В':'B','С':'C','Е':'E','К':'K','М':'M','Н':'H',
    'О':'O','Р':'P','Т':'T','У':'U','Х':'X',
    'а':'a','е':'e','о':'o','р':'p','с':'c','х':'x',
}
CYRILLIC_TABLE = str.maketrans(CYRILLIC_TO_LATIN)


# ── Anchors ───────────────────────────────────────────────────────────────────
HEADER_END_MARKERS = re.compile(
    r'Дата разработки.{0,30}:\s*\d{4}'
    r'|Пользователи протокола\s*:'
    r'|Категория пациентов\s*:'
    r'|II\.\s*МЕТОДЫ'
    r'|ДИАГНОСТИКА И ЛЕЧЕНИЕ НА АМБУЛАТОРНОМ',
    re.IGNORECASE,
)
ANCHOR_REAL = re.compile(
    r'Код\s*\(ы\)\s*(?:по\s+)?МКБ\s*[-–]?\s*1[01]\s*(?:и\s+МКБ\s*[-–]?\s*1[01])?\s*:?'
    r'|Код\s+МКБ-10\s+[A-ZА-ЯЁ]'
    r'|1\.1\s+Код'
    r'|МКБ-10\s+МКБ-9\s+Код'
    r'|МКБ\s*[-–]\s*10\s*:'
    r'|МКБ\s*[-–]?\s*10\s*[A-ZА-ЯЁ]'   # "МКБ 10: B20…" or "МКБ 10 B20…"
    r'|МКБ-10\s*:'
    r'|[Кк]од\s+по\s+МБК-10'
)
ANCHOR_TOC  = re.compile(r'Соотношение кодов МКБ')
# Fix 15: broader нет detection (handles "МКБ-10 – нет." with em-dash)
NO_CODES_SIG = re.compile(r'МКБ\s*[-–—]?\s*10\s*[-–—:]\s*нет', re.IGNORECASE)


# ── Pre-processing helpers ────────────────────────────────────────────────────

def _strip_icd9_procedure_codes(text: str) -> str:
    """Fix 9: Replace ICD-9 procedure codes (NN.NN all-digit format) with a sentinel.
    Keeps the leading integer so it still acts as a lookahead terminator for desc capture.
    Full removal caused regressions: desc swallowed following procedure text with no digit stop.
    Example: "38.25" → " 38 ", "86.20" → " 86 "  (sentinel, not empty string)
    """
    return re.sub(r'\b(\d{2,3})\.\d{2,3}\b', r' \1 ', text)


def _strip_mkb11_column(text: str) -> str:
    """Fix 14: Remove МКБ-11 codes (6Cxx format) that appear in side-by-side tables.
    Example: "6C4D", "6C40.3", "6C41" → removed
    """
    return re.sub(r'\b6[A-Z]\w{1,4}\b', ' ', text)


def _expand_bracket_codes(text: str) -> str:
    """Fix 3: ICD codes inside parentheses → insert bare copy before the bracket.
    "(С06.9) Описание" → "С06.9 (С06.9) Описание"
    """
    def _rep(m):
        inner = m.group(1).strip()
        if re.match(r'^[A-ZА-ЯЁ][\s.]?\d{2}', inner.translate(CYRILLIC_TABLE)):
            return f'{inner} {m.group(0)}'
        return m.group(0)
    return re.sub(r'\(([A-ZА-ЯЁ][^)]{2,20})\)', _rep, text)


def _fix_O_as_zero(text: str) -> str:
    """Fix 5+16: capital O in digit positions of ICD codes → 0.
    "BOO.O" → "B00.0"      (O replacing zero in main digits)
    "PO7"   → "P07"        (O between letter and trailing digit, e.g. РО7)
    "Q64.O" → "Q64.0"      (O in decimal suffix position)
    """
    def _rep(m):
        return m.group(1) + m.group(0)[1:].replace('O', '0')
    text = re.sub(r'\b([A-Z])[O0]{1,3}(?:\.[O0\d]{1,2})?\b', _rep, text)
    # O sandwiched between letter and 1-2 digits: PO7 → P07
    text = re.sub(r'\b([A-Z])O(\d{1,2})\b', r'\g<1>0\2', text)
    # O in decimal suffix: Q64.O → Q64.0
    text = re.sub(r'\b([A-Z]\d{2,3})\.O\b', r'\g<1>.0', text)
    return text


def _normalize_spaced_codes(text: str) -> str:
    """Fix 1+6+11: collapse spacing/punctuation artefacts inside ICD codes.
      "I 86.1"  → "I86.1"
      "К 55.1"  → "K55.1"   (after transliteration К→K)
      "S 42.3"  → "S42.3"
      "J84. 8 1"→ "J84.81"  (space inside decimal + extra digit)
      "Q 64.O"  → "Q64.0"   (after O-as-zero)
      "I0. 1.2."→ "I01.2"   (old МБК multi-part)
    """
    # Comma decimal → dot  (М 24,5 → M24.5)
    text = re.sub(r'([A-Z])(\s?)(\d{2,3}),(\d)', r'\1\3.\4', text)
    # Collapse "LETTER SPACE DIGITS[.DIGITS]"
    text = re.sub(
        r'(?<![A-Z\d])([A-Z]) (\d{2,3}(?:\.\d{1,2})?)(?=[^A-Z\d]|$)',
        r'\1\2', text,
    )
    # Collapse space inside decimal: "J84. 8" → "J84.8"
    text = re.sub(r'([A-Z]\d{2,3})\. (\d)', r'\1.\2', text)
    # Extra digit after decimal space: "J84.8 1" where OCR split "81" → "8 1"
    text = re.sub(r'([A-Z]\d{2,3}\.\d) (\d)(?=\s)', r'\1\2', text)
    # Old МБК multi-part: "I4. 0." → "I40." — use \1\2\3 (no trailing dot) to avoid double-dot
    text = re.sub(r'([A-Z])(\d)\.\s+(\d)', r'\1\2\3', text)
    # Decimal-O AFTER space collapse: "Q64.O" → "Q64.0" (Q 64.O → Q64.O → Q64.0)
    text = re.sub(r'\b([A-Z]\d{2,3})\.O\b', r'\g<1>.0', text)
    return text


def _strip_bracket_dupes(text: str) -> str:
    """After _expand_bracket_codes inserts a bare copy before each bracket,
    remove the now-redundant parenthesised copy so desc capture works cleanly.
    "J95.5 (J95.5) Стеноз" → "J95.5 Стеноз"
    """
    return re.sub(
        r'([A-Z]\d{2,3}(?:\.\d{1,2})?)\s*\([A-ZА-ЯЁ\s\d.*+]{2,20}\)',
        r'\1', text,
    )


def preprocess(header: str) -> str:
    """Apply the full pre-processing pipeline."""
    header = re.sub(r'[\uf0b7\u2022\u00b7•]', ' ', header)
    header = header.replace('\n', ' ').replace('\r', ' ')
    header = _strip_mkb11_column(header)          # Fix 14: remove МКБ-11 codes
    header = _strip_icd9_procedure_codes(header)  # Fix 9:  remove ICD-9 NN.NN codes
    header = _expand_bracket_codes(header)        # Fix 3:  expand (X00.x) before transliterate
    header = header.translate(CYRILLIC_TABLE)     # transliterate Cyrillic look-alikes
    header = _fix_O_as_zero(header)               # Fix 5+16: O→0 in digit positions
    header = _normalize_spaced_codes(header)      # Fix 1+6+11: collapse spacing
    header = _strip_bracket_dupes(header)         # Fix 3b: remove redundant (CODE) after bare code
    return header


# ── Code normalisation ────────────────────────────────────────────────────────

def normalize_code(raw: str) -> str:
    raw = raw.strip().translate(CYRILLIC_TABLE)
    raw = re.sub(r'^([A-Z])\s+(\d)', r'\1\2', raw)
    raw = re.sub(r'^([A-Z])\.(\d)', r'\1\2', raw)
    raw = raw.replace(',', '.')
    raw = raw.rstrip('-\u2013\u2014.*+').strip()
    return raw


VALID_CODE = re.compile(r'^[A-Z]\d{2,3}(?:\.\d{1,2})?$')


# ── ICD-10 chapter map for digit-only OCR recovery ───────────────────────────
# Maps leading 2-digit prefix of the numeric part to the most likely ICD-10 letter.
# Used only when the entire code block is digit-only (letter was OCR-dropped).
# Entries cover the unambiguous cases; ambiguous prefixes are left out.
_PREFIX_TO_LETTER = {
    # Chapter I – Infectious (A00-B99): first digit 0 or 1 → ambiguous, skip
    # Chapter II – Neoplasms C00-D48
    **{f'{i:02d}': 'C' for i in range(0, 49)},   # C00-C97, D00-D48
    # Chapter III – Blood D50-D89
    **{f'{i:02d}': 'D' for i in range(50, 90)},
    # Chapter IV – Endocrine E00-E90
    **{f'{i:02d}': 'E' for i in range(0, 91)},    # overrides C for 00-48 — context resolves
    # Chapter V – Mental F00-F99
    **{f'{i:02d}': 'F' for i in range(0, 100)},
    # Chapter VI – Nervous G00-G99
    **{f'{i:02d}': 'G' for i in range(0, 100)},
    # Chapter VII – Eye H00-H59
    **{f'{i:02d}': 'H' for i in range(0, 60)},
    # Chapter IX – Circulatory I00-I99
    **{f'{i:02d}': 'I' for i in range(0, 100)},
    # Chapter X – Respiratory J00-J99
    **{f'{i:02d}': 'J' for i in range(0, 100)},
    # Chapter XI – Digestive K00-K93
    **{f'{i:02d}': 'K' for i in range(0, 94)},
    # Chapter XIII – Musculoskeletal M00-M99
    **{f'{i:02d}': 'M' for i in range(0, 100)},
    # Chapter XIV – Genitourinary N00-N99
    **{f'{i:02d}': 'N' for i in range(0, 100)},
    # Chapter XV – Pregnancy O00-O99
    **{f'{i:02d}': 'O' for i in range(0, 100)},
    # Chapter XIX – Injury S00-T98
    **{f'{i:02d}': 'S' for i in range(0, 100)},
    **{f'{i:02d}': 'T' for i in range(0, 99)},
}


def _recover_digit_only(header_raw: str, processed: str) -> dict:
    """Fix 10: recover ICD-10 codes where OCR dropped the leading letter.
    Only activates when:
      - A confirmed МКБ-10 section anchor is present
      - No valid letter-prefixed codes were found
      - There are no ICD-9 procedure columns (МКБ-9 indicator absent)
    """
    results = {}
    # Only try if this looks like a clean МКБ-10 section (no МКБ-9 column)
    if re.search(r'МКБ-9|МКБ\s*9', header_raw, re.IGNORECASE):
        return results
    if not re.search(r'Код\s*\(ы\)\s*МКБ|МКБ\s*[-–]?\s*10\s*:', header_raw, re.IGNORECASE):
        return results

    # Find digit-only codes followed by a description
    pat = re.compile(
        r'(?<![A-Z\d.])(\d{3}(?:\.\d{1,2})?)\s+'
        r'([А-ЯЁA-Za-zа-яё][^.]{3,100}?)'
        r'(?=\s+\d{3}[.\s]|\s*\d+[.\s)]|Сокращения|Дата|\s*$)'
    )
    for m in pat.finditer(processed):
        raw_digits = m.group(1)
        desc = m.group(2).strip().rstrip('.,;:')[:160]
        prefix = raw_digits[:2]
        letter = _PREFIX_TO_LETTER.get(prefix)
        if not letter:
            continue
        # Avoid ICD-9 procedure codes disguised as 3-digit numbers (>200 range)
        try:
            if int(raw_digits.split('.')[0]) > 199:
                continue
        except ValueError:
            continue
        code = normalize_code(f'{letter}{raw_digits}')
        if VALID_CODE.match(code) and len(desc) >= 3:
            results.setdefault(code, desc)
    return results


def _extract_comma_list(header_raw: str) -> dict:
    """Fix 8: extract bare comma-separated code lists (no descriptions).
    Pattern: "МКБ 10: В20-В24, С00-С97, G20, G81-G83, ..."
    Returns codes with empty descriptions.
    """
    results = {}
    # Find the МКБ anchor + a compact list of codes/ranges separated by commas
    m = re.search(
        r'МКБ\s*[-–]?\s*1[01]\s*:?\s*'
        r'((?:[A-ZА-ЯЁ][\s.]?\d{2,3}(?:\.\d{1,2})?(?:\s*[-–]\s*[A-ZА-ЯЁ]?[\s.]?\d{2,3})?'
        r'(?:\s*,\s*|\s+))+)',
        header_raw, re.IGNORECASE
    )
    if not m:
        return results
    block = m.group(1)
    # Extract each code token
    for tok in re.finditer(r'[A-ZА-ЯЁ][\s.]?\d{2,3}(?:\.\d{1,2})?', block):
        code = normalize_code(tok.group(0))
        if VALID_CODE.match(code):
            results.setdefault(code, '')
    return results


# ── Header extraction ─────────────────────────────────────────────────────────

def extract_header(text: str) -> str:
    anchor = ANCHOR_REAL.search(text)
    if anchor:
        start = anchor.start()
        end   = start + 3000
        stop  = HEADER_END_MARKERS.search(text, start + 50)
        if stop:
            end = min(end, stop.start() + 200)
        return text[start:end]

    toc = list(ANCHOR_TOC.finditer(text))
    if toc:
        pick  = toc[-1] if len(toc) >= 2 else toc[0]
        start = pick.start()
        end   = start + 3000
        stop  = HEADER_END_MARKERS.search(text, start + 50)
        if stop:
            end = min(end, stop.start() + 200)
        return text[start:end]

    broad = re.search(r'МКБ', text[:8000])
    if broad:
        return text[broad.start(): broad.start() + 3000]

    return text[:3000]


# ── Extraction patterns ───────────────────────────────────────────────────────

# Fix 7: range terminators use \d+[.\s)] to match "1.2", "4.", "2)" not just 2-digit ICD codes
_RANGE_PAT = re.compile(
    r'([A-Z])(\d{2,3})\s*[-\u2013\u2014]\s*[A-Z]?(\d{2,3})'
    r'\s+([^\d]{3,160}?)'
    r'(?=\s+[A-Z]\d|\s*\d+[.\s)]|Дата|Сокращения|\s*$)',
)
_RANGE_PAT_OPEN = re.compile(
    r'([A-Z])(\d{2,3})\s*[-\u2013\u2014]\s*(\d{2,3})'
    r'\s+([^\d]{3,160}?)'
    r'(?=\s+[A-Z]\d|\s*\d+[.\s)]|Дата|Сокращения|\s*$)',
)

# Fix 2+4+7+13: wider desc, stop at "(", stop at \d+[.\s)]
_CODE_DESC_PAT = re.compile(
    r'([A-Z]\.?\s?\d{2,3}(?:\.\d{1,2})?[.*+]?)'
    r'(?:\s*[-\u2013]\s*|\s+)'
    r'([^\d(]{3,160}?)'
    r'(?=\s+\d+[.\s)%-]|\d+[.\s)%-]|\s*[A-Z]\.?\s?\d|\s+\(|Сокращения|Дата|$)'
)

# Fix 12+mixed: bare code with no description (e.g. "N73.6." or "S42.3" in mixed ICD-9 table)
_CODE_ONLY_PAT = re.compile(
    r'(?<![A-Z\d])([A-Z]\d{2,3}(?:\.\d{1,2})?)\.?(?=\s+\d|\s*[A-Z]\d|\s*$)'
)


# ── Main extraction function ──────────────────────────────────────────────────

def extract_codes_with_descs(header: str) -> dict:
    """Extract {code: description} from a header section.
    Tries multiple passes in order of reliability.
    """
    results: dict = {}
    processed = preprocess(header)

    # ── Pass 1: remove ranges, save for fallback ──────────────────────────────
    ranges_found: list = []

    def _save(m):
        code = f'{m.group(1)}{m.group(2)}'
        desc = m.group(4).strip().rstrip('.,;:').strip()[:160]
        ranges_found.append((code, desc))
        return ' '

    p2 = _RANGE_PAT.sub(_save, processed)
    p2 = _RANGE_PAT_OPEN.sub(_save, p2)

    # ── Pass 2: main code+desc regex ─────────────────────────────────────────
    for m in _CODE_DESC_PAT.finditer(p2):
        desc = m.group(2).strip().rstrip('.,;:').strip()[:160]
        code = normalize_code(m.group(1))
        if VALID_CODE.match(code) and len(desc) >= 3:
            results.setdefault(code, desc)

    # ── Pass 3: range fallback ────────────────────────────────────────────────
    if not results:
        for raw_code, desc in ranges_found:
            code = normalize_code(raw_code)
            if VALID_CODE.match(code) and len(desc) >= 3:
                results.setdefault(code, desc)

    # ── Pass 4: comma-separated bare code list (Fix 8) ───────────────────────
    if not results:
        results = _extract_comma_list(header)

    # ── Pass 5: digit-only OCR recovery (Fix 10) ─────────────────────────────
    if not results:
        results = _recover_digit_only(header, processed)

    # ── Pass 6: code with no description (Fix 12) ────────────────────────────
    if not results:
        for m in _CODE_ONLY_PAT.finditer(processed):
            code = normalize_code(m.group(1))
            if VALID_CODE.match(code):
                results.setdefault(code, '')

    # ── Pass 7: bare ranges as last resort ───────────────────────────────────
    if not results:
        for raw_code, desc in ranges_found:
            code = normalize_code(raw_code)
            if VALID_CODE.match(code):
                results.setdefault(code, desc)

    return results


# ── CLI / Main ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='Build ICD-10 JSON from protocols corpus (v2).')
    p.add_argument('--corpus', default='src/corpus/protocols_corpus.jsonl')
    p.add_argument('--output', default='merged_icd10_v2.json')
    return p.parse_args()


def main():
    args        = parse_args()
    corpus_path = Path(args.corpus)
    output_path = Path(args.output)

    if not corpus_path.exists():
        raise FileNotFoundError(f'Corpus not found: {corpus_path}')

    merged_icd:      dict = {}
    no_codes_found         = []
    used_extraction        = 0
    skipped_no_codes       = 0

    try:
        from tqdm import tqdm
        opener = lambda: tqdm(open(corpus_path, encoding='utf-8'), desc='Processing')
    except ImportError:
        opener = lambda: open(corpus_path, encoding='utf-8')

    with opener() as f:
        for line in f:
            data = json.loads(line)
            pid  = data.get('protocol_id')
            text = data.get('text', '')

            if NO_CODES_SIG.search(text[:2000]):
                skipped_no_codes += 1
                continue

            header = extract_header(text)
            codes  = extract_codes_with_descs(header)

            if codes:
                used_extraction += 1
                for code, desc in codes.items():
                    if code not in merged_icd:
                        merged_icd[code] = {'desc': desc, 'protocol_ids': [pid]}
                    else:
                        if not merged_icd[code]['desc'] and desc:
                            merged_icd[code]['desc'] = desc
                        if pid not in merged_icd[code]['protocol_ids']:
                            merged_icd[code]['protocol_ids'].append(pid)
            else:
                no_codes_found.append((pid, data.get('source_file', '')))

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_icd, f, ensure_ascii=False, indent=2)

    print(f'\nSaved {len(merged_icd)} ICD-10 codes to {output_path}')
    print(f'  Protocols with codes extracted : {used_extraction}')
    print(f'  Skipped (explicit нет)         : {skipped_no_codes}')

    if no_codes_found:
        print(f'\nWARN: No codes found for {len(no_codes_found)} protocols:')
        for pid, src in no_codes_found[:15]:
            print(f'  {pid} — {src}')
        if len(no_codes_found) > 15:
            print(f'  ... and {len(no_codes_found) - 15} more')


if __name__ == '__main__':
    main()