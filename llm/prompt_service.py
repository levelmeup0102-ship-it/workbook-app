"""프롬프트 로딩 + 인메모리 캐시 (DB: prompt_templates).

- 프롬프트는 모든 지문에 동일 → 첫 조회만 DB, 이후 캐시 재사용 (N요청 = DB 1회).
- ⚠ 캐시 무효화 함수 없음(의도적). 프롬프트 수정 시 서버 재시작으로 갱신.

    from llm.prompt_service import get_prompt
"""
import logging
from string import Template

from supabase import AsyncClient

from database import repository
from core.exceptions import PromptRenderError
from core.settings import settings
from database.repository import get_active_prompt

logger = logging.getLogger(__name__)

# 시스템 프롬프트 (env 저장) — 반드시 Railway env 설정 필요
SYS_JSON = settings.SYS_JSON
SYS_JSON_KR = settings.SYS_JSON_KR

# grammar_points → 시스템프롬프트 부착 텍스트 (step5 어법용). None=아직 미로드
_grammar_addendum_cache: str | None = None

# 프롬프트(LLM)가 필요한 step만. step7(스크램블)·step8(답안지조립)은 LLM 없음 → 제외
STEPS = [
    "step1_basic", "step2_order", "step3_blank", "step4_topic",
    "step5_grammar", "step6_vocab_content",
]

def render_prompt(template: str, **variables) -> str:
    """
    DB 프롬프트 템플릿($변수)을 실제 값으로 치환 (string.Template).
    JSON 중괄호 { } 는 그대로 유지됨. 누락 변수 있으면 KeyError → 조기 발견.
    """
    try:
        return Template(template).substitute(**variables)
    except KeyError as error:
        missing_key = error.args[0]
        raise PromptRenderError(message=f"프롬프트 변수 누락: {missing_key}") from error

async def get_prompt_from_database(client: AsyncClient | None, reading_round_number: int = 1) -> dict:
    """해당 회독 활성 프롬프트 조회 → {step: prompt_template} (STEPS 전 키 포함, 없으면 None).
    미등록 step은 경고 로깅. (캐시 없음 → 프롬프트 DB 수정이 다음 요청에 즉시 반영)
    """
    rows = await repository.get_active_prompt(client, reading_round_number)
    db_map = {r["step_name"]: r["prompt_template"] for r in rows if r.get("step_name")}
    prompts = {step: db_map.get(step) for step in STEPS}
    missing = [step for step in STEPS if not prompts[step]]
    if len(missing) > 0:
        logger.warning("프롬프트 미등록 step: %s (해당 step 생성 실패 예상)", missing)
    return prompts


def _format_grammar_points(rows: list) -> str:
    """grammar_points 행들을 시스템프롬프트 부착용 텍스트로 조립.
    (구 pipeline._load_grammar_points_for_prompt 의 조립부 원본 복사 — 로직 수정 없음)
    """
    if not rows:
        return ""

    # 카테고리별로 그룹화
    by_cat: dict = {}
    for r in rows:
        cat = r.get("category", "기타")
        by_cat.setdefault(cat, []).append(r)

    lines = [
        "═" * 60,
        "## 📚 한국 수능·내신 빈출 어법 함정 (반드시 우선 적용)",
        "═" * 60,
        "",
        "본문 분석 시 다음 어법 포인트를 우선적으로 발견하라.",
        "특히 '★ trap_warning'과 '🚫 prohibited_analysis'는 절대 위반 금지.",
        "지문에 trigger_keywords가 보이면 즉시 해당 어법으로 분석할 것.",
        "",
    ]
    star_map = {5: "★★★★★", 4: "★★★★", 3: "★★★", 2: "★★", 1: "★"}

    for cat, items in by_cat.items():
        lines.append(f"### [{cat}]")
        for it in items:
            star = star_map.get(it.get("priority", 3), "★★★")
            name = it.get("name", "")
            pattern = it.get("pattern") or ""
            ex_good = it.get("example_good") or ""
            ex_bad = it.get("example_bad") or ""
            trap = it.get("trap_warning") or ""
            prohibited = it.get("prohibited_analysis") or ""
            keywords = it.get("trigger_keywords") or []
            notes = it.get("notes") or ""

            lines.append(f"- **{star} {name}**")
            if pattern:
                lines.append(f"  - 패턴: {pattern}")
            if ex_good:
                lines.append(f"  - [O] {ex_good}")
            if ex_bad:
                lines.append(f"  - [X] {ex_bad}")
            if trap:
                lines.append(f"  - ⚠️ {trap}")
            if prohibited:
                lines.append(f"  - 🚫 금지: {prohibited}")
            if keywords:
                kw_str = ", ".join(keywords[:8])
                lines.append(f"  - 🔑 trigger: {kw_str}")
            if notes:
                lines.append(f"  - 💡 {notes}")
        lines.append("")

    lines.extend([
        "═" * 60,
        "## 적용 규칙 (RECAP)",
        "═" * 60,
        "1. 위 어법 중 본문 trigger_keywords에 해당하는 것은 무조건 grammar_notes에 포함",
        "2. prohibited_analysis 위반 절대 금지 (잘못된 해설 방지)",
        "3. 생략구문 분석 시 생략된 부분을 [I was], [it is], [have] 같이 대괄호로 명시",
        "4. 의미상 주어 + 동명사 패턴 (someone V-ing)은 단순 분사가 아닌 동명사로 분석",
        "5. 한국 수능·내신 빈출 어법 (도치/의미상주어/대동사/조동사 등) 우선 포함",
    ])

    return "\n".join(lines)


async def get_grammar_addendum(client: AsyncClient | None) -> str:
    """step5 어법 시스템프롬프트에 붙일 grammar_points 텍스트 반환 (캐시).
    모든 지문에 동일 → 첫 조회만 DB, 이후 캐시. 실패/미설정 시 "" (안전).
    """
    global _grammar_addendum_cache
    if _grammar_addendum_cache is None:
        rows = await repository.get_grammar_points(client)
        _grammar_addendum_cache = _format_grammar_points(rows)
        logger.info("[grammar_points 조회] DB 로드 → 캐시 (%d행, %d chars)",
                    len(rows), len(_grammar_addendum_cache))
    return _grammar_addendum_cache