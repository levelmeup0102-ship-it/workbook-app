"""
seosul 패키지 — 레벨미업학원 워크북 2회독 서술형 종합문제 모듈

main.py에 추가:
    from seosul.api import router as seosul_router
    app.include_router(seosul_router)

환경 변수 (variation 모듈과 동일):
    ANTHROPIC_API_KEY, CLAUDE_MODEL, SUPABASE_URL, SUPABASE_SERVICE_KEY, APP_PASSWORD

DB 의존:
    passages(지문) / seosul_types(유형 사양) / grammar_points(어법 화이트리스트·블랙리스트)

설계 원칙:
    LLM 단발 생성에 의존하지 않는다. 문장 역할 배정(코드) → 유형별 생성 → 검증 → 재생성.
    어법은 grammar_points 화이트리스트로만 출제(즉석 판단 금지), 블랙리스트는 자동 제외.
"""
__version__ = "1.0.0"

from seosul.api import router
from seosul.generator import generate_set
from seosul.renderer import render_seosul_set, render_pdf
from seosul.validator import validate_set

__all__ = ["router", "generate_set", "render_seosul_set", "render_pdf", "validate_set"]
