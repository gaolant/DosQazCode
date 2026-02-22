"""
Step 1: Build vector index from protocols_corpus.jsonl
Run once: python build_index.py --corpus protocols_corpus.jsonl
"""
import argparse
import json
from pathlib import Path
import re
import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split long protocol text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    step = max_chars - 200  # 200-char overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + max_chars]
        if chunk:
            chunks.append(chunk)
    return chunks


def load_protocol_to_icd(icd_json_path: Path) -> dict[str, list[str]]:
    """Build inverted index: protocol_id → list of ICD-10 codes."""
    with open(icd_json_path, "r", encoding="utf-8") as f:
        icd_db = json.load(f)
    mapping: dict[str, list[str]] = {}
    for code, entry in icd_db.items():
        for pid in entry.get("protocol_ids", []):
            mapping.setdefault(pid, []).append(code)
    return mapping


def build_index(corpus_path: Path, db_path: str, icd_json_path: Path | None):
    print(f"Loading corpus from {corpus_path}...")
    protocols = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                protocols.append(json.loads(line))
    print(f"Loaded {len(protocols)} protocols")

    # Load ICD lookup if provided
    protocol_to_icd: dict[str, list[str]] = {}
    if icd_json_path and icd_json_path.exists():
        protocol_to_icd = load_protocol_to_icd(icd_json_path)
        print(f"Loaded ICD-10 lookup: {len(protocol_to_icd)} protocols mapped")
    else:
        print("No ICD-10 JSON provided — will use codes from corpus only")

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        device="cuda",
    )

    client = chromadb.PersistentClient(path=db_path)
    try:
        client.delete_collection("protocols")
    except Exception:
        pass

    collection = client.create_collection(
        name="protocols",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metas = [], [], []
    chunk_idx = 0
    enriched = 0

    for protocol in protocols:
        protocol_id = protocol["protocol_id"]
        title       = protocol.get("title", "")
        source      = protocol.get("source_file", "")
        text        = protocol.get("text", "")

        # ── ICD-10 resolution ──────────────────────────────────────────────
        # Priority 1: codes already in the corpus entry
        icd_codes: list[str] = protocol.get("icd_codes", [])
        # Priority 2: fill gaps (or replace empty list) from your ICD JSON
        if not icd_codes and protocol_id in protocol_to_icd:
            icd_codes = protocol_to_icd[protocol_id]
            enriched += 1
        # ──────────────────────────────────────────────────────────────────

        text = re.split(
            r'Список использованной литературы|СПИСОК ЛИТЕРАТУРЫ'
            r'|Уровень доказательности|Классификация доказательств',
            text
        )[0]

        header    = f"Протокол: {source}\n{title}\nКоды МКБ-10: {', '.join(icd_codes)}\n\n"
        full_text = "passage: " + header + text
        chunks    = chunk_text(full_text, max_chars=1800)

        for i, chunk in enumerate(chunks):
            docs.append(chunk)
            ids.append(f"{protocol_id}_chunk_{i}")
            metas.append({
                "protocol_id": protocol_id,
                "icd_codes":   json.dumps(icd_codes, ensure_ascii=False),
                "source_file": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            chunk_idx += 1

        if len(docs) >= 50:
            collection.add(documents=docs, ids=ids, metadatas=metas)
            print(f"  Indexed {chunk_idx} chunks so far...")
            docs, ids, metas = [], [], []

    if docs:
        collection.add(documents=docs, ids=ids, metadatas=metas)

    print(f"Done! Total chunks indexed: {chunk_idx}")
    print(f"Protocols enriched with ICD JSON codes: {enriched}")
    print(f"ChromaDB saved to: {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True,  type=Path)
    parser.add_argument("--db",     default="./chroma_db")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--icd-json",
        type=Path,
        default=None,
        help="Path to icd10_codes.json (optional but recommended)",
    )
    args = parser.parse_args()
    build_index(args.corpus, args.db, args.icd_json)