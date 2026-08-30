"""Supabase 연결(Connection) 관리 — client 생성/정리만 담당 (쿼리는 repository).

lifespan 에서 사용:
    app.state.supabase = await init()   # startup
    await close(app.state.supabase)     # shutdown
"""
import logging

from supabase import acreate_client, AsyncClient

from core.settings import settings

logger = logging.getLogger(__name__)


async def init() -> AsyncClient | None:
    """env(SUPABASE_URL/KEY) 있으면 async client 생성, 없으면 None(로컬 모드)."""
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    if not (url and key):
        logger.warning("[supabase] SUPABASE_URL/SECRET_KEY 미설정 → 환경 변수 세팅 필요. 로컬 모드(client=None)")
        return None
    try:
        client = await acreate_client(url, key)
        logger.info("[supabase] async client 생성 완료(DB 연결 객체 생성)")
        return client
    except Exception as exc:
        logger.error("[supabase] 연결 실패 — URL 확인 필요: %s\n원인: %r", url, exc)
        raise RuntimeError(f"Supabase 연결 실패 (SUPABASE_URL={url}): {exc!r}") from exc
    

async def close(client: AsyncClient | None) -> None:
    if client is None:
        return

    try:
        await client.postgrest.aclose()
        logger.info("[supabase] postgrest client 종료 완료")
    except Exception as exc:
        logger.warning("[supabase] postgrest client 종료 중 예외(무시): %s", exc)
    