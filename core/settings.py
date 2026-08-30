"""환경설정 중앙 관리 — 모든 env 변수를 이 모듈 하나에서 로드.

각 파일은 os.environ 직접 접근 대신:
    from core.settings import settings
    settings.APP_PASSWORD

⚠️ import 순서 주의:
    테이블명(TBL_*)이 import 시점에 확정되므로 이 모듈은 repository 보다
    먼저 로드돼야 한다. → main.py 최상단에서 `from core.settings import settings`
    를 가장 먼저 import.

시크릿(SUPABASE/ANTHROPIC/APP_PASSWORD 등)은 OS 환경변수/배포 대시보드에서 직접 주입.
"""
import os


def _first(*names: str, default=None):
    """여러 env 이름 중 먼저 값이 있는 것 반환(별칭 호환용)."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default


class Settings:
    """전역 환경설정. import 시 1회 생성(기존 모듈-전역 로드와 동일 동작)."""

    def __init__(self) -> None:
        # ── 인증 (필수) ──
        self.APP_PASSWORD: str = os.environ["APP_PASSWORD"]

        # ── Supabase (서버 secret 키; 표준명=SUPABASE_SECRET_KEY, 나머지는 deprecated 별칭) ──
        self.SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
        self.SUPABASE_KEY: str | None = _first(
            "SUPABASE_SECRET_KEY",        # ← 표준(canonical)
            "SUPABASE_SERVICE_KEY",       # deprecated 별칭
            "SUPABASE_SERVICE_ROLE_KEY",  # deprecated 별칭
            "SUPABASE_KEY",               # deprecated 별칭
        )

        # ── LLM ──
        self.ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
        self.CLAUDE_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-5")
        self.SYS_JSON: str | None = os.environ.get("SYS_JSON")
        self.SYS_JSON_KR: str | None = os.environ.get("SYS_JSON_KR")

        # ── 환경/운영 ──
        self.ENV_DATA: str = os.environ.get("ENV_DATA", "DEV").upper()
        self.LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "WARNING").upper()
        self.PORT: int = int(os.environ.get("PORT", "8000"))

        # ── 출력 경로(레거시: variation/seosul) ──
        self.VARIATION_OUTPUT_DIR: str = os.environ.get("VARIATION_OUTPUT_DIR", "/tmp/variation_output")
        self.SEOSUL_OUTPUT_DIR: str = os.environ.get("SEOSUL_OUTPUT_DIR", "/tmp/seosul_output")

        # ── 테이블명 (ENV_DATA 분기; 구 core/config.py 흡수) ──
        if self.ENV_DATA == "PROD":
            self.TBL_PASSAGES: str = "passages"
            self.TBL_STEP_CACHE: str = "step_cache"
        else:
            self.TBL_PASSAGES = "passages_test"
            self.TBL_STEP_CACHE = "step_cache_test"
        self.TBL_PROMPTS: str = "prompt_templates"
        self.TBL_GRAMMAR: str = "grammar_points"

    @property
    def is_dev(self) -> bool:
        return self.ENV_DATA != "PROD"

    @property
    def supabase_enabled(self) -> bool:
        return bool(self.SUPABASE_URL and self.SUPABASE_KEY)


settings = Settings()
