# PII Redactor

Upload a PDF (scanned or searchable), review detected PII page-by-page, and save
masked images plus structured PII records to a local SQLite database.

- **Backend**: FastAPI + SQLModel/SQLite + Pillow + PyMuPDF (no Poppler needed)
- **Vision (OCR with bboxes)**: `qwen2.5vl:7b` via Ollama
- **PII tagging (text)**: `qwen3:8b` via Ollama
- **Frontend**: SvelteKit (Svelte 5) per-page review UI with manual edit/draw

## Pipeline

For each page:

1. PDF page is rasterized to PNG (PyMuPDF, configurable DPI).
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
| `VL_BACKEND` | `ollama` | `ollama` or `llamacpp` (upstream llama-server, for models Ollama hasn't wired vision for). |
| `LLAMACPP_HOST` | `http://localhost:11500` | llama-server base URL (when `VL_BACKEND=llamacpp`). |
| `VL_PROMPT_MODE` | `qwen` | `qwen` (Qwen2.5-VL prompt) or `dotsocr` (dots.ocr's native `prompt_layout_all_en`). |
| `DATABASE_URL` | `sqlite:///./storage/pii.db` | SQLAlchemy URL |
| `STORAGE_DIR` | `./storage` | Local file storage root |
| `PDF_DPI` | `150` | Rasterization DPI |
| `VL_MAX_PIXELS` | `802816` | Cap total pixels sent to VL (matches Qwen's smart_resize). Larger = better OCR, more VRAM. Upstream default is 1003520. |
| `VL_NUM_CTX` | `0` | Context window for VL. `0` = use Ollama's Modelfile default (safest — custom values can trigger GGML_ASSERT on some builds). |
| `VL_NUM_PREDICT` | `0` | Max new tokens for VL. `0` = Ollama default. |
| `PII_NUM_CTX` | `0` | Context window for Qwen3 PII call. `0` = Ollama default. |
| `PII_NUM_PREDICT` | `0` | Max new tokens for Qwen3. `0` = Ollama default. |
| `REFINE_BBOXES` | `true` | Tighten VL bboxes to actual text content (PIL threshold). |
| `REFINE_EXPAND_PCT` | `0.10` | Expand VL bbox by this % before tightening (recovers missing edges). |
| `REFINE_THRESHOLD` | `180` | Grayscale threshold: pixels darker than this count as text. |
| `MASK_PADDING_PX` | `2` | Extra pixels around each bbox when drawing the black mask. |

## Using dots.ocr via llama-server

Ollama doesn't have vision wiring for `dots.ocr` yet (the main GGUF loads as a
text-only qwen2 model). Run it through upstream llama.cpp's `llama-server`
instead:

```bash
# Download both GGUFs from https://huggingface.co/ggml-org/dots.ocr-GGUF
#   dots.ocr-f16.gguf (or another quant)
#   mmproj-dots.ocr-Q8_0.gguf (the vision projector)

# macOS:
brew install llama.cpp

llama-server \
  -m ~/models/dots-ocr/dots.ocr-f16.gguf \
  --mmproj ~/models/dots-ocr/mmproj-dots.ocr-Q8_0.gguf \
  --port 11500 --host 127.0.0.1
```

Then in `backend/.env`:
```
VL_BACKEND=llamacpp
LLAMACPP_HOST=http://localhost:11500
VL_PROMPT_MODE=dotsocr
VL_MODEL=dots-ocr     # any name; llama-server ignores it
```
Restart uvicorn; the PII tagger (Qwen3) keeps running on Ollama.

## Notes & limitations

- VL bboxes can be off; the UI lets you delete, edit type, or draw new boxes
  before confirming.
- If Qwen3 returns a PII string that doesn't appear verbatim in the VL chunks,
  it is dropped (no bbox to attach). This is intentional to keep masking
  faithful to what's visible.
- No authentication — intended for local use.
