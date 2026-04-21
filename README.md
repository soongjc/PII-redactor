# PII Redactor

Upload a PDF (scanned or searchable), review detected PII page-by-page, and save
masked images plus structured PII records to a local SQLite database.

- **Backend**: FastAPI + SQLModel/SQLite + Pillow + `pdf2image` (Poppler)
- **Vision (OCR with bboxes)**: `qwen2.5vl:7b` via Ollama
- **PII tagging (text)**: `qwen3:8b` via Ollama
- **Frontend**: SvelteKit (Svelte 5) per-page review UI with manual edit/draw

## Pipeline

For each page:

1. PDF page is rasterized to PNG (`pdf2image`, configurable DPI).
2. Qwen2.5-VL receives the image and returns `{text, bbox}` chunks (JSON mode).
3. Qwen3 receives the joined text and returns `{type, text}` PII spans.
4. Each PII text is matched back to one or more consecutive chunks; the union
   bounding box is returned to the UI for review.
5. User edits selections (toggle / change type / delete / draw new boxes), then
   confirms the page. The backend renders a masked PNG (black rectangles) and
   writes the entities to SQLite.
6. After every page is confirmed, "Finalize" combines the masked PNGs into a
   single `masked.pdf`.

## Data model (SQLite)

- `document(id, filename, page_count, status, created_at, finalized_at)`
- `page(id, document_id, page_number, width, height, confirmed, confirmed_at)`
- `piientity(id, page_id, type, text, x, y, w, h)`

Files on disk:

```
backend/storage/
  uploads/<doc_id>.pdf
  pages/<doc_id>/page_<n>.png
  masked/<doc_id>/page_<n>.png
  masked/<doc_id>/masked.pdf
  pii.db
```

## Prerequisites

- Python 3.10+
- Node 18+
- [Poppler](https://poppler.freedesktop.org/) on `PATH` (required by `pdf2image`)
  - macOS: `brew install poppler`
  - Debian/Ubuntu: `sudo apt-get install poppler-utils`
- [Ollama](https://ollama.com/) running locally with the two models pulled:
  ```bash
  ollama pull qwen2.5vl:7b
  ollama pull qwen3:8b
  ```

## Run

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # adjust if needed
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The Vite dev server proxies `/api/*` to
`http://localhost:8000`.

## API

- `POST /api/documents` — multipart upload (`file=<pdf>`); returns `{id, page_count, ...}`
- `GET  /api/documents/{id}` — document + per-page state
- `GET  /api/documents/{id}/pages/{n}/image` — original page PNG
- `POST /api/documents/{id}/pages/{n}/detect` — runs VL + PII, returns entities with bboxes
- `POST /api/documents/{id}/pages/{n}/confirm` — body `{entities: [{type,text,x,y,w,h}, ...]}`; saves to DB and renders masked PNG
- `GET  /api/documents/{id}/pages/{n}/masked` — masked page PNG
- `POST /api/documents/{id}/finalize` — requires every page confirmed; combines into `masked.pdf`
- `GET  /api/documents/{id}/masked.pdf` — download finalized masked PDF

## Configuration

Backend settings (env vars or `backend/.env`):

| Variable | Default | Notes |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `VL_MODEL` | `qwen2.5vl:7b` | Vision-language model tag |
| `PII_MODEL` | `qwen3:8b` | PII tagger model tag |
| `DATABASE_URL` | `sqlite:///./storage/pii.db` | SQLAlchemy URL |
| `STORAGE_DIR` | `./storage` | Local file storage root |
| `PDF_DPI` | `150` | Rasterization DPI |

## Notes & limitations

- VL bboxes can be off; the UI lets you delete, edit type, or draw new boxes
  before confirming.
- If Qwen3 returns a PII string that doesn't appear verbatim in the VL chunks,
  it is dropped (no bbox to attach). This is intentional to keep masking
  faithful to what's visible.
- No authentication — intended for local use.
