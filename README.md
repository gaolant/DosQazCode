# MedAssist KZ — Диагностический ассистент

AI-powered medical diagnosis assistant based on official clinical protocols of the Republic of Kazakhstan (МЗ РК). Built for the QazCode hackathon.

---

## How it works

1. User describes patient symptoms in Russian
2. Symptoms are normalized into clinical terms via LLM
3. Relevant protocols are retrieved from a vector index (ChromaDB)
4. LLM selects the top 5 most likely ICD-10 diagnoses from the retrieved protocols

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build ICD-10 codes JSON

```bash
python build_icd_json_v2.py --corpus protocols_corpus.jsonl --output merged_icd10_v2.json
```

### 3. Build vector index

```bash
python build_index_new.py --corpus protocols_corpus.jsonl --icd-json merged_icd10_v2.json
```

### 4. Start the server

```bash
uvicorn server_new:app --host 0.0.0.0 --port 8000
```

### 5. Open the UI

Open `index.html` in your browser (double-click or drag into Chrome/Firefox).

The UI connects to the server running on `localhost:8000` — make sure the server is running before opening the page.

---

## Docker

```bash
docker build -t medassist-kz .
docker run -p 8000:8000 \
  -e LLM_API_KEY=your_key \
  -e LLM_HUB_URL=your_url \
  -v $(pwd)/chroma_db:/app/chroma_db \
  medassist-kz
```

Then open `index.html` in your browser.

---

## API

### `POST /diagnose`

```json
{
  "symptoms": "болит голова у ребёнка, температура 38"
}
```

Response:
```json
{
  "diagnoses": [
    {
      "rank": 1,
      "icd10_code": "G43.0",
      "diagnosis": "Мигрень без ауры",
      "explanation": "..."
    }
  ]
}
```

### `GET /health`

Returns server status and number of indexed protocols.

---

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `LLM_API_KEY` | API key for LLM | `YOUR_API_KEY` |
| `LLM_HUB_URL` | LLM endpoint URL | `YOUR_HUB_URL` |
| `CHROMA_DB_PATH` | Path to ChromaDB | `./chroma_db` |
| `ICD_JSON_PATH` | Path to ICD-10 JSON | `./merged_icd10_v2.json` |

---

## Project structure

```
├── server_new.py          # FastAPI server
├── build_index_new.py     # Vector index builder
├── build_icd_json_v2.py   # ICD-10 extractor
├── evaluate.py            # Evaluation script
├── index.html             # Chat UI
├── Dockerfile
├── requirements.txt
└── README.md
```
