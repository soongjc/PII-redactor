from typing import Optional
from pydantic import BaseModel


class EntityIn(BaseModel):
    type: str
    text: str
    x: int
    y: int
    w: int
    h: int


class EntityOut(EntityIn):
    id: Optional[int] = None


class DetectResponse(BaseModel):
    page_number: int
    width: int
    height: int
    entities: list[EntityOut]


class ConfirmRequest(BaseModel):
    entities: list[EntityIn]


class PageOut(BaseModel):
    page_number: int
    width: int
    height: int
    confirmed: bool


class DocumentOut(BaseModel):
    id: int
    filename: str
    page_count: int
    status: str
    pages: list[PageOut] = []
