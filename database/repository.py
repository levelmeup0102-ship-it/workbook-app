"""Supabase 쿼리(Data Access) — 구 supa.py 를 SDK + DI 로 재구현.

- `client`(AsyncClient | None) 를 주입받는다 (DI): 모든 함수 첫 인자로 받아 lifespan으로 관리되고 있는 DB 연결 객체 사용.
- `client is None`(로컬 모드)이면 쿼리 없이 기본값 반환 → @_guard 데코레이터가 일괄 처리
  (기존 supa._request 가 not _enabled 시 None 반환하던 동작 보존).
- 인프라(_enabled/_headers/_request/모듈 전역 URL·KEY)는 제거 (client 가 대체).

확정 매핑: docs/refactor_plan.md 참고.
"""
import logging
import math
from typing import Any, List, Dict
from functools import wraps

from supabase import AsyncClient

from core.settings import settings
from core.exceptions import NotFoundError

logger = logging.getLogger(__name__)

# 테이블명은 core.settings 에서 주입 (dev=passages_test / prod=passages 등)
PASSAGES = settings.TBL_PASSAGES
STEP_CACHE = settings.TBL_STEP_CACHE
PROMPTS = settings.TBL_PROMPTS
GRAMMAR = settings.TBL_GRAMMAR


def _guard(default):
    """client is None(로컬 모드)이면 쿼리 없이 default 반환.
    default 가 callable(list 등)이면 호출해 새 인스턴스 반환, 아니면 그대로.
    """
    def deco(fn):
        @wraps(fn)
        async def wrapper(client, *args, **kwargs):
            if client is None:
                return default() if callable(default) else default
            return await fn(client, *args, **kwargs)
        return wrapper
    return deco


# ========================
# Passages
# ========================
async def _fetch_all_rows(client: AsyncClient, table: str, columns: str) -> List[Dict[str, Any]]:
    """지정 테이블/컬럼의 전체 행을 페이지네이션으로 조회 (PostgREST 기본 1000행 제한 회피).

    - 먼저 전체 행 수(count=exact)를 구한 뒤 PAGE_SIZE 단위로 range 조회해 누적.
    - table/columns 를 인자로 받아 호출마다 독립적으로 조회 → 테이블 간 데이터 혼입 없음.
    - client is None(로컬 모드) 처리는 호출부의 @_guard 가 담당.
    """
    PAGE_SIZE = 1000

    # 전체 행 개수 구하기 (head=True → 데이터 없이 count 만)
    count_res = await (
        client.table(table)
        .select("*", count="exact", head=True)
        .execute()
    )
    total_count = count_res.count or 0
    total_pages = math.ceil(total_count / PAGE_SIZE)

    rows: List[Dict[str, Any]] = []
    for page in range(total_pages):
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE - 1
        res = await (
            client.table(table)
            .select(columns)
            .range(start, end)
            .execute()
        )
        rows.extend(res.data or [])
    return rows


@_guard(default=list)
async def get_all_passages(client: AsyncClient) -> List[Dict[str, Any]]:
    """전체 지문 조회 (1000행 제한 회피 — _fetch_all_rows 로 전량 페이지네이션)."""
    return await _fetch_all_rows(client, PASSAGES, "book,unit,pid,title,passage_text")


@_guard(default=None)          # None시 None
async def get_passage(client: AsyncClient, book, unit, pid) -> Dict:
    """단일 지문 조회"""
    res = await (
        client.table(PASSAGES).select("*")
        .eq("book", book).eq("unit", unit).eq("pid", pid)
        .execute()
    )
    
    if not res.data:
        raise NotFoundError("요청한 지문이 DB에 없습니다.")
    
    row = res.data[0]

    return row


@_guard(default=list) # 교재 None -> return []
async def get_passages_by_book(client: AsyncClient, book) -> List:
    """
    특정 교재의 지문 전체 조회 -> 지문 목록 반환
    주 사용: delete_book 의 404 판정 + 지문별 캐시 정리용 — cache_key컬럼 값은 지문마다 _ck()로 계산)
    """
    res = await client.table(PASSAGES).select("*").eq("book", book).execute()
    return res.data


@_guard(default=None)
async def upsert_passage(client: AsyncClient, book, unit, pid, title, text) -> List:
    """지문 업데이트"""
    row = {
        "book": book,
        "unit": unit,
        "pid": pid,
        "title": title,
        "passage_text": text
        }
    res = await client.table(PASSAGES).upsert(row, on_conflict="book,unit,pid").execute()
    return res.data


@_guard(default=None)
async def upsert_passages_bulk(client: AsyncClient, rows: list) -> List:
    """대량의 지문 업데이트"""
    if not rows:
        return None
    logger.info("[supabase] upserting %d passages...", len(rows))
    res = await client.table(PASSAGES).upsert(rows, on_conflict="book,unit,pid").execute()
    logger.info("[supabase] UPSERT success: %d rows", len(res.data) if res.data else 0)
    return res.data


# ========================
# Prompt Templates (table: prompt_templates)
# ========================
@_guard(default=None)
async def get_active_prompt(client: AsyncClient, reading_round_number: int = 1) -> List[dict]:
    """활성(is_active) 프롬프트 템플릿 1개 조회. 없으면 None.
    (활성 버전은 부분 유니크 인덱스로 (round, step_key) 당 1개 보장)
    """
    res = await (
        client.table(PROMPTS).select("step_name, prompt_template")
        .eq("reading_round_number", reading_round_number)
        .eq("is_active", True)
        .execute()
    )
    return res.data


# ========================
# Grammar Points (table: grammar_points) — step5 어법 시스템프롬프트 주입용
# ========================
_GRAMMAR_POINT_COLUMNS = (
    "category,subcategory,priority,name,pattern,"
    "example_good,example_bad,why_important,trap_warning,"
    "prohibited_analysis,trigger_keywords,notes"
)


@_guard(default=list)
async def get_grammar_points(client: AsyncClient) -> List:
    """활성 어법 포인트(priority>=3) 조회 — priority DESC, category ASC.
    (구 _load_grammar_points_for_prompt 의 httpx fetch 를 SDK 로 대체)
    """
    res = await (
        client.table(GRAMMAR).select(_GRAMMAR_POINT_COLUMNS)
        .eq("active", True)
        .gte("priority", 3)
        .order("priority", desc=True)
        .order("category")
        .execute()
    )
    return res.data


# ========================
# Step Cache (table: step_cache)
# ========================
@_guard(default=None)
async def get_step(client: AsyncClient, cache_key, step_name) -> Dict:
    """step_cache 행 반환: {"data": <결과 JSONB>, "passage_hash": <text>} 또는 None.
    (해시 검증은 호출측 cache 계층에서 — repository 는 행만 반환)
    """
    res = await (
        client.table(STEP_CACHE).select("data, passage_hash")
        .eq("cache_key", cache_key).eq("step_name", step_name)
        .execute()
    )
    return res.data[0] if res.data else None


@_guard(default=None)
async def save_step(client: AsyncClient, cache_key, step_name, data: dict, passage_hash: str) -> List:
    """구 save_step_supa. step_cache 에 upsert. data(순수 결과) + passage_hash(별도 컬럼)."""
    row = {
        "cache_key": cache_key,
        "step_name": step_name,
        "data": data,
        "passage_hash": passage_hash,
    }
    res = await client.table(STEP_CACHE).upsert(row, on_conflict="cache_key,step_name").execute()
    return res.data


@_guard(default=0)             # None시 0
async def count_steps(client: AsyncClient, cache_key) -> int:
    """cache_key 에 캐시된 step 개수 (>=8 이면 워크북 생성 완료로 간주)."""
    res = await client.table(STEP_CACHE).select("step_name").eq("cache_key", cache_key).execute()
    return len(res.data) if isinstance(res.data, list) else 0


@_guard(default=dict)          # None시 {}
async def count_steps_all(client: AsyncClient) -> Dict:
    """전체 step_cache 를 조회 → cache_key별 step 개수 dict 반환.
    (list_passages 의 지문별 count_steps N+1 을 배치로 대체 + 1000행 제한 회피)
    """
    rows = await _fetch_all_rows(client, STEP_CACHE, "cache_key")
    counts: dict = {}
    for row in rows:
        ck = row.get("cache_key")
        if ck:
            counts[ck] = counts.get(ck, 0) + 1
    return counts


@_guard(default=None)
async def delete_steps_by_cache_key(client: AsyncClient, cache_key) -> List:
    """cache_key 의 모든 step 캐시 삭제."""
    res = await client.table(STEP_CACHE).delete().eq("cache_key", cache_key).execute()
    return res.data


@_guard(default=None)
async def delete_step(client: AsyncClient, cache_key, step_name) -> List:
    """단일 step 캐시 삭제."""
    res = await (
        client.table(STEP_CACHE).delete()
        .eq("cache_key", cache_key).eq("step_name", step_name)
        .execute()
    )
    return res.data


async def delete_all_steps(client: AsyncClient | None, cache_key) -> List:
    """delete_steps_by_cache_key 별칭 (pipeline.py 호환)."""
    return await delete_steps_by_cache_key(client, cache_key)


# ========================
# Delete passages
# ========================
@_guard(default=None)
async def delete_passage(client: AsyncClient, book, unit, pid) -> List:
    """교재의 지문 삭제"""
    res = await (
        client.table(PASSAGES).delete()
        .eq("book", book).eq("unit", unit).eq("pid", pid)
        .execute()
    )
    return res.data


@_guard(default=None)
async def delete_book(client: AsyncClient, book) -> List:
    """교재 삭제(CASCADE)"""
    res = await client.table(PASSAGES).delete().eq("book", book).execute()
    return res.data
