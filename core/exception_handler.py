"""예외 → 통일 JSONResponse 변환 핸들러.

main 에서 등록:
    from core.exceptions import AppError
    from core.exception_handler import app_error_handler, unhandled_handler
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_handler)

응답 형식: {"ok": False, {"code": <머신용>, "error": <메시지>}}
"""
import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.exceptions import AppError

logger = logging.getLogger(__name__)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """AppError(+서브클래스) → 정의된 status_code/code 로 응답."""
    if exc.status_code >= 500:
        logger.exception("[%s] Application Error - code: %s, path: %s\nmessage: %s", exc.status_code, exc.code, request.url.path, exc.message)
    else:
        logger.warning("[%s] %s @ %s: %s", exc.status_code, exc.code, request.url.path, exc.message, exc_info=exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": exc.code,
                "message": exc.message
                }
            }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    
    logger.warning(
        "[StarletteHTTPException %s] %s @ %s",
        exc.status_code,
        request.url.path,
        exc.detail
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "code": f"HTTP_{exc.status_code}",
                "message": repr(exc)[:200]
            }
        }
    )


async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    
    logger.warning("[422] REQUEST_VALIDATION_ERROR @ %s: %s", request.url.path, exc.errors())

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "error": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "요청하는 값이 올바르지 않습니다."
            }
        }
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """예상 못 한 예외 → 500. 서버엔 전체 트레이스백 / 클라이언트엔 타입+요약(200자)."""
    import traceback as tb

    logger.exception("Unhandled error @ %s: %s", request.method, request.url.path)
    
    error_message = repr(exc) or "No error message"

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": {
                "code": "internal_error",
                "message": f"{type(exc).__name__}: {error_message}"
                }
            }
    )
