"""books 도메인 서비스 — 교재 전체 삭제 비즈니스 로직.

원본(main.py:394-424)의 로컬 dict 순회를 DB 조회(repository)로 교체.
"""
from core.exceptions import BadRequestError, NotFoundError
from database import repository as repo
from utils.cache_key import _ck

from .schemas import DeleteBookIn


async def delete_book(payload: DeleteBookIn, client) -> dict:
    """교재 존재 확인(없으면 404) → 하위 지문 캐시 일괄 정리 → 교재 삭제.

    get_passages_by_book 조회를 404 판정과 캐시 정리에 함께 활용(방법 C):
    지문 목록으로 각 지문의 cache_key(_ck)를 정확히 계산해 캐시를 지운다.
    """
    book = (payload.book or "").strip()
    if not book:
        raise BadRequestError("book 필요")

    # 404 판정 + 캐시 정리 대상 확보 (조회 1회를 두 용도로 재사용)
    passages = await repo.get_passages_by_book(client, book)
    if not passages:
        raise NotFoundError("book not found")

    # 하위 지문들의 step 캐시 정리 (고아 데이터 방지)
    for p in passages:
        await repo.delete_steps_by_cache_key(client, _ck(p.get("book"), p.get("unit"), p.get("pid")))

    # 교재(지문 전체) 삭제
    await repo.delete_book(client, book)
    return {"ok": True}
