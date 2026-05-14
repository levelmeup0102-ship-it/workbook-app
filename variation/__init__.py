"""
variation 패키지 — 레벨미업학원 워크북 2회독 변형문제 모듈

main.py에 다음 2줄을 추가하세요:
    from variation.api import router as variation_router, download_router
    app.include_router(variation_router)
    app.include_router(download_router)

환경 변수:
    ANTHROPIC_API_KEY   — Claude API 키 (필수)
    SUPABASE_URL        — Supabase URL (기존 사용 중)
    SUPABASE_SERVICE_KEY 또는 SUPABASE_KEY — Supabase 키 (기존 사용 중)
    APP_PASSWORD        — 인증 토큰 (기존 main.py와 동일, 기본값 levelmeup2026)
    CLAUDE_MODEL        — Claude 모델 (기본 claude-sonnet-4-5-20251022)
    VARIATION_OUTPUT_DIR — PDF 임시 저장 경로 (기본 /tmp/variation_output)
"""

__version__ = "1.0.0"

from variation.api import router, download_router
from variation.generator import generate_variation_a, generate_variation_b
from variation.renderer import render_variation_pdf

__all__ = [
    "router",
    "download_router",
    "generate_variation_a",
    "generate_variation_b",
    "render_variation_pdf",
]
