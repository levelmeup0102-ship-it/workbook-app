"""books 라우터 — DELETE /api/books.

prefix="/api/books" + dependencies=[Depends(verify)].
"""
from fastapi import APIRouter, Depends, Request

from core.security import _verify

from . import service
from .schemas import DeleteBookIn

router = APIRouter(prefix="/api/books", tags=["books"], dependencies=[Depends(_verify)])


@router.delete(
        "",
        summary="교재 단위로 삭제",
        description=""
        )
async def delete_book(payload: DeleteBookIn, request: Request):
    """DELETE /api/books: 교재 단위로 삭제하는 API입니다."""
    return await service.delete_book(payload, request.app.state.supabase)
