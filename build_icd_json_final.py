import json
import re
from tqdm import tqdm

CORPUS_JSONL = "src/corpus/protocols_corpus.jsonl"
OUTPUT_JSON  = "merged_icd10_corpus_updated.json"

CYRILLIC_TO_LATIN = {
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'К': 'K',
    'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'Т': 'T',
    'У': 'U', 'Х': 'X',
}
CYRILLIC_TABLE = str.maketrans(CYRILLIC_TO_LATIN)

HEADER_END_MARKERS = re.compile(
    r'Дата разработки.{0,30}:\s*\d{4}'
    r'|Пользователи протокола\s*:'
    r'|Категория пациентов\s*:'
    r'|II\.\s*МЕТОДЫ'
    r'|ДИАГНОСТИКА И ЛЕЧЕНИЕ НА АМБУЛАТОРНОМ',
    re.IGNORECASE,
)

ANCHOR_REAL = re.compile(
    r'Код\(ы\)\s*МКБ-10\s*:'
    r'|Код\s+МКБ-10\s+[A-ZА-ЯЁ]'
    r'|1\.1\s+Код'
    r'|МКБ-10\s+МКБ-9\s+Код'
    r'|МКБ\s*[–\-]\s*10\s*:'
    r'|МКБ-10\s*:'
)

ANCHOR_TOC = re.compile(r'Соотношение кодов МКБ')
NO_CODES_SIGNAL = re.compile(r'МКБ-10\s*:\s*нет', re.IGNORECASE)


def normalize_code(raw: str) -> str:
    raw = raw.strip().translate(CYRILLIC_TABLE)
    raw = re.sub(r'^([A-Z])\s+(\d)', r'\1\2', raw)
    raw = re.sub(r'^([A-Z])\.(\d)', r'\1\2', raw)
    raw = raw.rstrip('-–—.*+').strip()
    return raw


def extract_header(text: str) -> str:
    anchor = ANCHOR_REAL.search(text)
    if anchor:
        start = anchor.start()
        end   = start + 2000
        section_end = HEADER_END_MARKERS.search(text, start + 50)
        if section_end:
            end = min(end, section_end.start() + 200)
        return text[start:end]

    toc_matches = list(ANCHOR_TOC.finditer(text))
    if toc_matches:
        m = toc_matches[-1] if len(toc_matches) >= 2 else toc_matches[0]
        start = m.start()
        end   = start + 2000
        section_end = HEADER_END_MARKERS.search(text, start + 50)
        if section_end:
            end = min(end, section_end.start() + 200)
        return text[start:end]

    broad = re.search(r'МКБ', text[:8000])
    if broad:
        return text[broad.start():broad.start() + 2000]

    return text[:5000]


def extract_codes_with_descs(header: str) -> dict[str, str]:
    results: dict[str, str] = {}

    header = re.sub(r'[\uf0b7\u2022\u00b7•]', ' ', header)
    header = header.replace('\n', ' ').replace('\r', ' ')

    # Collect ranges for fallback
    ranges_found = re.findall(
        r'([A-ZА-ЯЁ]\.?\s*\d{2,3})\s*[-–—]\s*[A-ZА-ЯЁ]?\.?\s*\d{2,3}'
        r'\s+([^\d]{3,80}?)(?=\s+[A-ZА-ЯЁ]\.?\s?\d|\s*\d\d|\s*$)',
        header
    )

    header_clean = re.sub(
        r'[A-ZА-ЯЁ]\.?\s*\d{2,3}\s*[-–—]\s*[A-ZА-ЯЁ]?\.?\s*\d{2,3}',
        '', header
    )

    # Lookahead stops at:
    # - another ICD-like code (letter + digit)
    # - a numbered section like "4." or "1.2" (digit+dot pattern, NOT mid-word)
    # - end of string
    pattern = re.compile(
        r'([A-ZА-ЯЁ]\.?\s?\d{2,3}(?:\.\d{1,2})?[.*+]?)'
        r'(?:\s*[–\-]\s*|\s+)'
        r'([^\d]{3,80}?)'
        r'(?=\s+\d+[.\s]|\s*[A-ZА-ЯЁ]\.?\s?\d|$)'
    )

    for match in pattern.finditer(header_clean):
        raw_code = match.group(1)
        desc     = match.group(2).strip().rstrip('.,;:').strip()[:120]
        code     = normalize_code(raw_code)

        if re.match(r'^[A-Z]\d{2,3}(?:\.\d{1,2})?$', code) and len(desc) >= 3:
            if code not in results:
                results[code] = desc

    # Fallback: use range start codes if nothing specific found
    if not results:
        for raw_code, desc in ranges_found:
            desc = desc.strip().rstrip('.,;:').strip()[:120]
            code = normalize_code(raw_code)
            if re.match(r'^[A-Z]\d{2,3}(?:\.\d{1,2})?$', code) and len(desc) >= 3:
                if code not in results:
                    results[code] = desc

    return results


# ── Main ──────────────────────────────────────────────────────────────────────
merged_icd: dict = {}
no_codes_found   = []
used_extraction  = 0
skipped_no_codes = 0

with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing corpus"):
        data = json.loads(line)
        pid  = data.get("protocol_id")
        text = data.get("text", "")

        if NO_CODES_SIGNAL.search(text[:2000]):
            skipped_no_codes += 1
            continue

        header = extract_header(text)
        codes  = extract_codes_with_descs(header)

        if codes:
            used_extraction += 1
            for code, desc in codes.items():
                if code not in merged_icd:
                    merged_icd[code] = {"desc": desc, "protocol_ids": [pid]}
                else:
                    if not merged_icd[code]["desc"] and desc:
                        merged_icd[code]["desc"] = desc
                    if pid not in merged_icd[code]["protocol_ids"]:
                        merged_icd[code]["protocol_ids"].append(pid)
        else:
            no_codes_found.append((pid, data.get("source_file", "")))

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(merged_icd, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(merged_icd)} ICD-10 codes to {OUTPUT_JSON}")
print(f"  Protocols with codes extracted : {used_extraction}")
print(f"  Skipped (explicit нет)         : {skipped_no_codes}")

if no_codes_found:
    print(f"\nWARN: No codes found for {len(no_codes_found)} protocols:")
    for pid, src in no_codes_found[:10]:
        print(f"  {pid} — {src}")
    if len(no_codes_found) > 10:
        print(f"  ... and {len(no_codes_found) - 10} more")