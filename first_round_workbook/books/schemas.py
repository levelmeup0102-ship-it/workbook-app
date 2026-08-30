"""books 도메인 요청 모델."""
from pydantic import BaseModel


class DeleteBookIn(BaseModel):
    """DELETE /api/books 요청 (main.py:364)."""
    book: str
