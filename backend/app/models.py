from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field, Relationship


class Document(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    page_count: int
    status: str = Field(default="pending")  # pending | in_review | finalized
    created_at: datetime = Field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = None

    pages: list["Page"] = Relationship(back_populates="document")


class Page(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: int = Field(foreign_key="document.id", index=True)
    page_number: int
    width: int
    height: int
    confirmed: bool = Field(default=False)
    confirmed_at: Optional[datetime] = None

    document: Optional[Document] = Relationship(back_populates="pages")
    entities: list["PIIEntity"] = Relationship(back_populates="page")


class PIIEntity(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    page_id: int = Field(foreign_key="page.id", index=True)
    type: str
    text: str
    x: int
    y: int
    w: int
    h: int

    page: Optional[Page] = Relationship(back_populates="entities")
