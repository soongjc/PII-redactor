from datetime import datetime
from pathlib import Path
import json

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select

from ..config import settings
from ..db import get_session
from ..models import Document, Page, PIIEntity
from ..pdf_utils import rasterize_pdf
from ..pipeline import detect_page_pii_stream
from ..masking import render_masked, combine_pdf
from ..schemas import (
    ConfirmRequest,
    DetectResponse,
    DocumentOut,
    EntityOut,
    PageOut,
)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _doc_pages_dir(doc_id: int) -> Path:
    return settings.pages_dir / str(doc_id)


def _doc_masked_dir(doc_id: int) -> Path:
    return settings.masked_dir / str(doc_id)


def _page_image(doc_id: int, n: int) -> Path:
    return _doc_pages_dir(doc_id) / f"page_{n}.png"


def _masked_image(doc_id: int, n: int) -> Path:
    return _doc_masked_dir(doc_id) / f"page_{n}.png"


@router.post("", response_model=DocumentOut)
async def upload(file: UploadFile = File(...), session: Session = Depends(get_session)) -> DocumentOut:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    doc = Document(filename=file.filename, page_count=0, status="pending")
    session.add(doc)
    session.commit()
    session.refresh(doc)
    assert doc.id is not None

    pdf_path = settings.uploads_dir / f"{doc.id}.pdf"
    pdf_path.write_bytes(await file.read())

    try:
        pages_info = rasterize_pdf(pdf_path, _doc_pages_dir(doc.id))
    except Exception as exc:
        session.delete(doc)
        session.commit()
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Failed to read PDF: {exc}") from exc

    for page_number, w, h in pages_info:
        session.add(Page(document_id=doc.id, page_number=page_number, width=w, height=h))
    doc.page_count = len(pages_info)
    doc.status = "in_review"
    session.add(doc)
    session.commit()
    session.refresh(doc)

    pages = session.exec(select(Page).where(Page.document_id == doc.id).order_by(Page.page_number)).all()
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        page_count=doc.page_count,
        status=doc.status,
        pages=[PageOut(page_number=p.page_number, width=p.width, height=p.height, confirmed=p.confirmed) for p in pages],
    )


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, session: Session = Depends(get_session)) -> DocumentOut:
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pages = session.exec(select(Page).where(Page.document_id == doc_id).order_by(Page.page_number)).all()
    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        page_count=doc.page_count,
        status=doc.status,
        pages=[PageOut(page_number=p.page_number, width=p.width, height=p.height, confirmed=p.confirmed) for p in pages],
    )


@router.get("/{doc_id}/pages/{n}/image")
def get_page_image(doc_id: int, n: int) -> FileResponse:
    path = _page_image(doc_id, n)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Page image not found")
    return FileResponse(path, media_type="image/png")


@router.get("/{doc_id}/pages/{n}/masked")
def get_masked_image(doc_id: int, n: int) -> FileResponse:
    path = _masked_image(doc_id, n)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Masked page not found")
    return FileResponse(path, media_type="image/png")


@router.post("/{doc_id}/pages/{n}/detect")
def detect_page(doc_id: int, n: int, session: Session = Depends(get_session)):
    """Streams NDJSON stage events (vl_ocr → pii_tag → done)."""
    page = session.exec(
        select(Page).where(Page.document_id == doc_id, Page.page_number == n)
    ).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    image = _page_image(doc_id, n)
    if not image.exists():
        raise HTTPException(status_code=404, detail="Page image missing on disk")

    page_number = page.page_number
    width = page.width
    height = page.height

    def gen():
        try:
            for event in detect_page_pii_stream(image):
                if event.get("stage") == "done":
                    event = {
                        **event,
                        "page_number": page_number,
                        "width": width,
                        "height": height,
                    }
                yield json.dumps(event) + "\n"
        except Exception as exc:  # noqa: BLE001 - stream to UI instead of 500
            yield json.dumps({"stage": "error", "where": "pipeline", "message": str(exc)}) + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/{doc_id}/pages/{n}/confirm", response_model=DetectResponse)
def confirm_page(
    doc_id: int,
    n: int,
    body: ConfirmRequest,
    session: Session = Depends(get_session),
) -> DetectResponse:
    page = session.exec(
        select(Page).where(Page.document_id == doc_id, Page.page_number == n)
    ).first()
    if not page:
        raise HTTPException(status_code=404, detail="Page not found")
    assert page.id is not None

    # Replace entities for this page.
    existing = session.exec(select(PIIEntity).where(PIIEntity.page_id == page.id)).all()
    for e in existing:
        session.delete(e)
    saved: list[PIIEntity] = []
    for e in body.entities:
        row = PIIEntity(
            page_id=page.id, type=e.type, text=e.text, x=e.x, y=e.y, w=e.w, h=e.h
        )
        session.add(row)
        saved.append(row)

    # Render mask now so the user sees the result of THIS confirmation.
    render_masked(
        _page_image(doc_id, n),
        _masked_image(doc_id, n),
        body.entities,
        padding=settings.mask_padding_px,
    )

    page.confirmed = True
    page.confirmed_at = datetime.utcnow()
    session.add(page)
    session.commit()
    for row in saved:
        session.refresh(row)

    return DetectResponse(
        page_number=page.page_number,
        width=page.width,
        height=page.height,
        entities=[
            EntityOut(id=r.id, type=r.type, text=r.text, x=r.x, y=r.y, w=r.w, h=r.h)
            for r in saved
        ],
    )


@router.post("/{doc_id}/finalize", response_model=DocumentOut)
def finalize(doc_id: int, session: Session = Depends(get_session)) -> DocumentOut:
    doc = session.get(Document, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    pages = session.exec(
        select(Page).where(Page.document_id == doc_id).order_by(Page.page_number)
    ).all()
    if not pages:
        raise HTTPException(status_code=400, detail="Document has no pages")
    unconfirmed = [p.page_number for p in pages if not p.confirmed]
    if unconfirmed:
        raise HTTPException(
            status_code=400,
            detail=f"Pages not yet confirmed: {unconfirmed}",
        )

    masked_paths = [_masked_image(doc_id, p.page_number) for p in pages]
    missing = [str(p) for p in masked_paths if not p.exists()]
    if missing:
        raise HTTPException(status_code=500, detail=f"Missing masked files: {missing}")
    combine_pdf(masked_paths, _doc_masked_dir(doc_id) / "masked.pdf")

    doc.status = "finalized"
    doc.finalized_at = datetime.utcnow()
    session.add(doc)
    session.commit()
    session.refresh(doc)

    return DocumentOut(
        id=doc.id,
        filename=doc.filename,
        page_count=doc.page_count,
        status=doc.status,
        pages=[
            PageOut(page_number=p.page_number, width=p.width, height=p.height, confirmed=p.confirmed)
            for p in pages
        ],
    )


@router.get("/{doc_id}/masked.pdf")
def get_masked_pdf(doc_id: int) -> FileResponse:
    path = _doc_masked_dir(doc_id) / "masked.pdf"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Finalized PDF not found")
    return FileResponse(path, media_type="application/pdf", filename=f"document_{doc_id}_masked.pdf")
