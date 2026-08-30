"""인증 유틸 — 토큰 생성/검증.

main.py 와 first_round_workbook 패키지 양쪽이 import 하는 중립 위치.
    from core.security import _token, _verify
"""
import hashlib
import logging

from fastapi import Request, HTTPException

from core.settings import settings

logger = logging.getLogger(__name__)

APP_PASSWORD = settings.APP_PASSWORD


def _token(pw: str) -> str:
    return hashlib.sha256(f"{pw}_wb2026".encode()).hexdigest()[:32]


def _verify(r: Request) -> None:
    request_headers = r.headers.get("Authorization", "").replace("Bearer ", "")
    if request_headers != _token(APP_PASSWORD):
        logger.warning("인증 실패: 유효하지 않은 인증 토큰, path=%s", r.url.path)
        raise HTTPException(401)
