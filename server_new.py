"""
Main FastAPI server: POST /diagnose
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import json
import os
import re

import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


# ── Config ────────────────────────────────────────────────────────────────────
API_KEY  = os.environ.get("LLM_API_KEY", "YOUR_API_KEY")
HUB_URL  = os.environ.get("LLM_HUB_URL", "YOUR_HUB_URL")
MODEL    = "oss-120b"
DB_PATH  = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
ICD_PATH = os.environ.get("ICD_JSON_PATH", "./merged_icd10_v2.json")
TOP_K    = 5   # how many protocol chunks to retrieve
N_DIAG   = 5   # how many diagnoses to return

# ── Clients ───────────────────────────────────────────────────────────────────
llm_client = OpenAI(base_url=HUB_URL, api_key=API_KEY)

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-large",
    device="cuda",  # change to "cpu" if no GPU
)
chroma_client = chromadb.PersistentClient(path=DB_PATH)
collection = chroma_client.get_collection(name="protocols", embedding_function=ef)

# ── Load ICD-10 lookup ────────────────────────────────────────────────────────
with open(ICD_PATH, "r", encoding="utf-8") as f:
    _ICD_DB: dict = json.load(f)

# Inverted index: protocol_id → list of {code, desc}
_PROTOCOL_TO_ICD: dict[str, list[dict]] = {}
for _code, _entry in _ICD_DB.items():
    for _pid in _entry.get("protocol_ids", []):
        _PROTOCOL_TO_ICD.setdefault(_pid, []).append(
            {"code": _code, "desc": _entry.get("desc", "")}
        )

app = FastAPI(title="Medical Diagnosis API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response schemas ────────────────────────────────────────────────
class DiagnoseRequest(BaseModel):
    symptoms: str


class Diagnosis(BaseModel):
    rank: int
    icd10_code: str
    diagnosis: str
    explanation: str


class DiagnoseResponse(BaseModel):
    diagnoses: list[Diagnosis]


# ── Core logic ────────────────────────────────────────────────────────────────
def extract_icd_from_text(text: str) -> list[str]:
    """Extract ICD-10 codes from protocol text as last-resort fallback."""
    pattern = r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b'
    codes = re.findall(pattern, text)
    return list(dict.fromkeys(codes))


def normalize_query(symptoms: str) -> str:
    try:
        response = llm_client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты — медицинский ассистент. Извлеки из описания симптомов "
                        "краткий список клинических терминов на русском языке для поиска "
                        "по медицинским протоколам. Только термины через запятую, без пояснений."
                    ),
                },
                {"role": "user", "content": symptoms},
            ],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.choices[0].message.content
        if not content:
            return symptoms

        # Detect garbled output — if mostly non-Cyrillic/non-Latin, fall back
        clean_chars = sum(1 for c in content if c.isalpha())
        question_marks = content.count('?')
        if clean_chars == 0 or question_marks > len(content) * 0.3:
            print(f"[normalize_query] garbled output detected, using raw symptoms")
            return symptoms

        return content.strip()
    except Exception as e:
        print(f"[normalize_query] failed: {e}, falling back to raw symptoms")
        return symptoms


def retrieve_protocols(query: str) -> list[dict]:
    """
    Vector search + ICD enrichment.
    Priority: ICD JSON → ChromaDB metadata → regex fallback.
    """
    results = collection.query(
        query_texts=["query: " + query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    seen_protocols = set()
    SIM_THRESHOLD = 0.5

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if dist > SIM_THRESHOLD:
            continue

        pid = meta["protocol_id"]

        # ── ICD-10 resolution (three-tier) ─────────────────────────────────
        icd_entries: list[dict] = _PROTOCOL_TO_ICD.get(pid, [])
        if icd_entries:
            # Tier 1: authoritative codes + descriptions from ICD JSON
            icd_codes = [e["code"] for e in icd_entries]
        else:
            raw_codes = json.loads(meta["icd_codes"])
            if raw_codes:
                # Tier 2: codes from ChromaDB metadata (no descriptions)
                icd_codes = raw_codes
                icd_entries = [{"code": c, "desc": ""} for c in icd_codes]
            else:
                # Tier 3: regex extraction from chunk text
                icd_codes = extract_icd_from_text(doc)
                icd_entries = [{"code": c, "desc": ""} for c in icd_codes]
        # ───────────────────────────────────────────────────────────────────

        if pid not in seen_protocols:
            seen_protocols.add(pid)

        hits.append({
            "protocol_id": pid,
            "icd_codes":   icd_codes,
            "icd_entries": icd_entries,
            "source_file": meta["source_file"],
            "text_chunk":  doc,
            "distance":    dist,
        })

    return hits


def build_prompt(symptoms: str, protocols: list[dict]) -> tuple[str, str]:
    """Build the RAG prompt with protocol context and ICD-10 descriptions."""
    context_parts = []
    code_desc_map: dict[str, str] = {}  # deduplicated code → desc for allowed block

    for i, p in enumerate(protocols, 1):
        # Collect descriptions for allowed codes block
        for e in p["icd_entries"]:
            if e["code"] not in code_desc_map:
                code_desc_map[e["code"]] = e["desc"]

        # Format codes with inline descriptions for context
        codes_str = "; ".join(
            f"{e['code']} ({e['desc'][:60]})" if e["desc"] else e["code"]
            for e in p["icd_entries"]
        )
        chunk_preview = p["text_chunk"][:800]
        context_parts.append(
            f"--- Протокол {i} (источник: {p['source_file']}) ---\n"
            f"Коды МКБ-10: {codes_str}\n"
            f"Текст: {chunk_preview}\n"
        )

    context = "\n".join(context_parts)

    # Allowed codes block — includes descriptions so LLM can reason semantically
    allowed_block = "\n".join(
        f"  {code}: {desc[:80]}" if desc else f"  {code}"
        for code, desc in code_desc_map.items()
    )

    system_prompt = f"""Ты — клинический ассистент, работающий с официальными протоколами Республики Казахстан.

На основе жалоб пациента и предоставленных протоколов определи наиболее вероятные диагнозы.

ВАЖНЫЕ ПРАВИЛА:
1. Возвращай ТОЛЬКО коды МКБ-10 из списка допустимых кодов ниже.
2. Выбирай наиболее СПЕЦИФИЧНЫЙ код (например G91.1 лучше чем G91).
3. Ранжируй по вероятности — rank 1 самый вероятный.
4. Верни ровно {N_DIAG} диагнозов в JSON.
5. Отвечай ТОЛЬКО JSON, без лишнего текста.

Допустимые коды МКБ-10 (код: описание):
{allowed_block}

Формат ответа:
{{
  "diagnoses": [
    {{"rank": 1, "icd10_code": "X00.0", "diagnosis": "Название диагноза на русском", "explanation": "Краткое обоснование на основе симптомов и протокола"}},
    ...
  ]
}}"""

    user_prompt = f"""Жалобы пациента:
{symptoms}

Релевантные протоколы РК:
{context}

Определи {N_DIAG} наиболее вероятных диагнозов из допустимых кодов МКБ-10."""

    return system_prompt, user_prompt


def parse_llm_response(content: str) -> list[dict]:
    """Parse LLM JSON output, handling common formatting issues."""
    content = re.sub(r"```(?:json)?", "", content).strip().strip("`").strip()
    try:
        data = json.loads(content)
        return data["diagnoses"]
    except (json.JSONDecodeError, KeyError):
        match = re.search(r'\{.*"diagnoses".*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())["diagnoses"]
        raise ValueError(f"Could not parse LLM response: {content[:200]}")


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/debug")
async def debug(request: DiagnoseRequest):
    clinical_query = normalize_query(request.symptoms)
    protocols = retrieve_protocols(clinical_query)
    return {
        "clinical_query": clinical_query,
        "hits": [
            {
                "source":   p["source_file"],
                "codes":    p["icd_codes"][:5],
                "entries":  p["icd_entries"][:5],
                "distance": p["distance"],
            }
            for p in protocols
        ],
    }


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    if not request.symptoms.strip():
        raise HTTPException(status_code=400, detail="symptoms field is empty")

    # 1. Normalize free-form symptoms to clinical terms
    clinical_query = normalize_query(request.symptoms)

    # 2. Retrieve relevant protocols
    protocols = retrieve_protocols(clinical_query)
    if not protocols:
        raise HTTPException(status_code=500, detail="No protocols retrieved")

    # 3. Build prompt
    system_prompt, user_prompt = build_prompt(request.symptoms, protocols)

    # 4. Call LLM
    response = llm_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,  # low temp for deterministic medical output
        max_tokens=1500,
    )
    content = response.choices[0].message.content

    # 5. Parse and return
    try:
        raw_diagnoses = parse_llm_response(content)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    diagnoses = [
        Diagnosis(
            rank=d["rank"],
            icd10_code=d["icd10_code"],
            diagnosis=d.get("diagnosis", ""),
            explanation=d.get("explanation", ""),
        )
        for d in raw_diagnoses
    ]

    return DiagnoseResponse(diagnoses=diagnoses)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "collection_count": collection.count(),
        "icd_codes_loaded": len(_ICD_DB),
        "protocols_mapped": len(_PROTOCOL_TO_ICD),
    }