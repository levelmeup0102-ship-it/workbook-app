"""1회독 교재 생성 — 집계 라우터.

3개 도메인 라우터(passages/books/generation)를 묶어 하나로 노출.
main.py 는 이 router 하나만 include 한다:

    from first_round_workbook.router import router as first_round_router
    app.include_router(first_round_router)
"""
from fastapi import APIRouter

from .books.router import router as books_router
from .generation.router import router as generation_router
from .passages.router import router as passages_router

router = APIRouter(tags=["first_round_router: 1회독 교재 생성 API"])

router.include_router(generation_router)
router.include_router(books_router)
router.include_router(passages_router)
