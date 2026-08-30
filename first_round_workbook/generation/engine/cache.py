"""1회독 엔진 전용 step 캐시 — Supabase(step_cache)만 사용, 해시 기반 무효화.

- 캐시 유효성 = 저장된 _passage_hash 와 현재 지문 해시 일치 여부 (STEP_VERSIONS 미사용).
- 지문 재업로드로 내용이 바뀌면 해시가 달라져 자동 무효화 → 재생성.
- 기존 캐시(_passage_hash 필드 없음)는 해시 불일치로 간주 → 재생성(옵션 A).
- DB 접근은 repository 를 통해 client(DI) 주입.
"""
import logging

from supabase import AsyncClient

from database import repository
from utils.cache_key import _hash

logger = logging.getLogger(__name__)


async def load_cache_if_passage_unchanged(
    client: AsyncClient | None,
    cache_key: str,
    step_name: str,
    passage_text: str
) -> dict | None:
    """
    1. 캐시 우선 조회: 지문이 안 바뀐 경우(해시 일치)에만 캐시 결과 반환.
    1-1. 반환된 결과가 없거나 해시 불일치(=지문 변경/기존 캐시)면 None → 결과 재생성.
    """
    row = await repository.get_step(client, cache_key, step_name)
    if not row:
        return None
    if row.get("passage_hash") != _hash(passage_text):
        return None
    return row.get("data")


async def save_step_result(
    client: AsyncClient | None,
    cache_key: str,
    step_name: str,
    result: dict,
    passage_text: str,
) -> None:
    """생성 결과 저장 — data(순수 결과) + passage_hash(별도 컬럼) 함께 저장."""
    await repository.save_step(client, cache_key, step_name, result, _hash(passage_text))


async def load_cache_result(
    client: AsyncClient | None,
    cache_key: str,
    step_name: str
) -> dict | None:
    """폴백용(조건: 재시도 3회 실패) — step_cache 에 저장된 결과가 있으면 그대로 반환(이전 결과 사용)."""
    row = await repository.get_step(client, cache_key, step_name)
    return row.get("data") if row else None