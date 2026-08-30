"""Claude 비동기 호출 — AsyncAnthropic 기반.

구 lib-async-claude.py 의 핵심 함수만 가져와 정리 (실험 코드 제외).
LLM 은 무상태라 모듈 전역 lazy 싱글톤으로 클라이언트 재사용 (DI 아님).

    from llm.client import call_claude_json_async, call_claude_file_async
"""
import asyncio
import logging

from anthropic import (
    AsyncAnthropic,
    AuthenticationError, PermissionDeniedError, NotFoundError,
    BadRequestError, RateLimitError, APITimeoutError,
    APIConnectionError, APIStatusError,
)

from core.exceptions import (
    LLMError, LLMConfigError, LLMRequestError,
    LLMRateLimitError, LLMTemporaryError, LLMTimeoutError,
)
from core.settings import settings
from utils.json_parser import parse_json_robust

logger = logging.getLogger(__name__)

MODEL = settings.CLAUDE_MODEL

_client: AsyncAnthropic | None = None


def _map_llm_error(e: Exception) -> LLMError:
    """Anthropic SDK 예외/기타 예외 → 앱 LLMError 계열로 변환 (재시도 여부는 클래스가 앎)."""
    if isinstance(e, LLMError):
        return e                                          # 이미 매핑됨
    if isinstance(e, (AuthenticationError, PermissionDeniedError, NotFoundError)):
        return LLMConfigError(str(e))                     # 모델명/키/권한/청구 → 재시도 X
    if isinstance(e, BadRequestError):
        return LLMRequestError(str(e))                    # payload/파라미터 오류 → 재시도 X
    if isinstance(e, RateLimitError):
        return LLMRateLimitError(str(e))                  # 429
    if isinstance(e, APITimeoutError):
        return LLMTimeoutError(str(e))
    if isinstance(e, APIConnectionError):
        return LLMTemporaryError(str(e))                  # 연결 오류 → 재시도
    if isinstance(e, APIStatusError):
        # 5xx(500/529 등) → 일시, 그 외 4xx → 요청 오류
        status = getattr(e, "status_code", 500)
        return LLMTemporaryError(str(e)) if status >= 500 else LLMRequestError(str(e))
    return LLMTemporaryError(str(e))                      # 미분류(JSON 파싱 실패 등) → 재시도


def _get_client() -> AsyncAnthropic:
    """lazy 싱글톤 — 첫 호출 시(실행 루프 안에서) 생성 후 재사용."""
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


# def _file_ids() -> list[str]:
#     """CLAUDE_FILE_IDS(콤마 구분) → 리스트."""
#     return [fid.strip() for fid in os.environ.get("CLAUDE_FILE_IDS", "").split(",") if fid.strip()]


async def call_claude_json_async(
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
    max_tokens: int = 4096
) -> dict:
    """일반 Claude 호출 → JSON 파싱 반환. 실패 시 지수 backoff 재시도."""
    for attempt in range(max_retries + 1):
        try:
            response = await _get_client().messages.create(
                model=MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            # 안전 거부(refusal): 같은 입력엔 계속 거부하므로 비재시도 처리
            # 설명: API 호출 자체는 성공했지만 Claude가 정책/안전 필터 등의 이유로 답변 생성을 거절한 상태
            if response.stop_reason == "refusal":
                raise LLMRequestError("모델이 안전상 응답을 거부함(stop_reason=refusal)")
            text = response.content[0].text.strip()
            return parse_json_robust(text)
        except Exception as e:
            err = _map_llm_error(e)
            # 비재시도(설정/요청 오류) 또는 마지막 시도 → 즉시 실패
            if not err.retryable or attempt >= max_retries:
                logger.error("[LLM] %s → 실패: %s", err.code, e)
                raise err from e
            logger.warning("[LLM] %s 재시도 %d/%d: %s", err.code, attempt + 1, max_retries, e)
            await asyncio.sleep(2 * (attempt + 1))

    # 루프는 항상 return/raise로 끝나지만, 타입 안정성 위해 명시적 종결
    raise LLMTemporaryError("LLM 호출 재시도 소진")


# async def call_claude_file_async(
#     system_prompt: str,
#     user_prompt: str,
#     max_retries: int = 3,
#     max_tokens: int = 4096,
# ) -> dict:
#     """Files API(문서 첨부) Claude 호출 → JSON 파싱 반환. (step5 어법: 금지유형 문서 참조)"""
#     documents = [
#         {"type": "document", "source": {"type": "file", "file_id": fid}}
#         for fid in _file_ids()
#     ]
#     last_error = None
#     for attempt in range(max_retries + 1):
#         try:
#             response = await _get_client().beta.messages.create(
#                 model=MODEL,
#                 max_tokens=max_tokens,
#                 betas=["files-api-2025-04-14"],
#                 system=system_prompt,
#                 messages=[{
#                     "role": "user",
#                     "content": documents + [{"type": "text", "text": user_prompt}],
#                 }],
#             )
#             text = response.content[0].text.strip()
#             return parse_json_robust(text)
#         except Exception as e:
#             last_error = e
#             if attempt < max_retries:
#                 await asyncio.sleep(2 * (attempt + 1))
#             else:
#                 raise ValueError(f"Claude file async call failed: {last_error}") from last_error
