"""앱 공통 예외 — base AppError + 도메인 4종.

핸들러(core.exception_handler)가 status_code/code/message 로 통일 JSON 생성.
서비스/라우터에서 raise 만 하면 되고, HTTP 변환은 핸들러가 담당.
"""

from fastapi import HTTPException

class AppError(Exception):
    """모든 앱 예외의 부모. status_code/code 를 서브클래스가 오버라이드."""
    status_code: int = 500
    code: str = "INTERNAL_SERVER_ERROR"
    message = "서버 내부 오류가 발생했습니다."

    # 에러 메시지만 받음.
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)
        


class BadRequestError(AppError):
    """잘못된 요청"""
    status_code = 400
    code = "BADREQUEST_ERROR"
    message = "잘못된 요청입니다."



class UnauthorizedError(AppError):
    """인증 실패."""
    status_code = 401
    code = "UNAUTHORIZED_ERROR"
    message = "인증이 필요합니다."


class ForbiddenError(AppError):
    """접근 권한 없음. 로그인 페이지로 리다이렉트"""
    status_code = 403
    code = "FORBIDDEN_ERROR"
    message = "접근 권한이 없습니다."


class NotFoundError(AppError):
    """요청한 리소스 없음. 혹은 존재하지 않는 리소스 요청"""
    status_code = 404
    code = "RESOURCE_NOT_FOUND_ERROR"
    message = "요청한 리소스를 찾을 수 없습니다."


class GenerationError(AppError):
    """생성/파이프라인 실패."""
    status_code = 500
    code = "WORKBOOK_GENERATION_ERROR"
    message = "교재 생성 중 오류가 발생했습니다."


# ========================
# LLM ERROR
# ========================
class PromptRenderError(GenerationError):
    """DB에서 가져온 프롬프트의 템플릿 렌더 실패(변수 누락/문법 오류) - 값 할당 등의 설정 오류"""
    code = "PROMPT_RENDER_ERROR"
    message="프롬프트 템플릿에 값이 제대로 할당되지 못했습니다."


# ── LLM 외부 호출 계열 (현재: Claude) ──
# retryable: 재시도가 의미 있는지 여부. client/orchestrator 가 이 값으로 재시도 분기.
class LLMError(GenerationError):
    code = "LLM_Error"
    message = "Claude 호출 중 오류 발생"
    retryable: bool = False

class LLMConfigError(LLMError):
    """모델명/API Key/권한/청구 - 재시도 로직에서 제외"""
    code = "LLM_CONFIG_ERROR"
    message = "LLM 설정 오류입니다. (환경변수 - 모델명/API Key/권한/청구(billing) 확인 필요)"
    retryable = False

class LLMRequestError(LLMError):
    """payload 구성 오류 / 요청 크기 초과 / 잘못된 파라미터 - 재시도 로직에서 제외"""
    code = "LLM_REQUEST_ERROR"
    message = "LLM 요청이 올바르지 않습니다. 잘못된 값으로 요청하였거나, 크기를 초과했습니다."
    retryable = False

class LLMRateLimitError(LLMError):
    """429 rate limit"""
    code = "LLM_RATE_LIMIT_ERROR"
    message = "LLM 요청이 한도를 초과했습니다. 나중에 다시 시도해주세요."
    retryable = False

class LLMTemporaryError(LLMError):
    """500/529/연결오류 - Claude 쪽 서버에서 내부 오류 발생"""
    code = "LLM_TEMPORARY_ERROR"
    message = "AI 서버나 내부의 일시적 오류입니다."
    retryable = True

class LLMTimeoutError(LLMError):
    """Timeout"""
    code = "LLM_TIMEOUT_ERROR"
    message = "LLM 응답 시간이 초과되었습니다"
    retryable = True