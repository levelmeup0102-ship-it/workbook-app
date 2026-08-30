"""1회독 워크북 생성 오케스트레이터.

- run_step: step 1개 처리 — 캐시 우선 → 생성(3회 재시도) → 폴백(이전 캐시) → 실패.
- generate_workbook: step1~8 을 의존 순서(단계: 0→1→2)대로 조율 + 결과 수집.

step 함수는 async 로 이식 예정 (engine/steps.py). 지금은 구조/시그니처 기준.
LLM 실호출은 Claude SDK 단계에서 step 함수 내부에 채움.
"""
import asyncio
import logging

from supabase import AsyncClient

from core.exceptions import GenerationError, LLMError
from . import cache
from . import steps
from . import answer_sheet
from . import render

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


async def run_step(
    client: AsyncClient | None,
    step_function,
    cache_key: str,
    step_name: str,
    passage_text: str,
    *step_args
) -> dict:
    """
    step 1개 실행.

    과정:
    - 캐시 우선 확인
    - passage_text_for_cache_compare로 현재 지문 hash 계산
    - 캐시의 passage_hash와 비교
    - 없거나 stale이면 생성
    - 생성 실패 시 이전 캐시 폴백

    Args:
        - client: Supabase client
        - step_function: 실제 실행할 step 함수
        - cache_key: 지문/교재 단위 캐시 키
        - step_name: step 식별 이름
        - passage_text: 지문 업데이트 유무 확인용 / 새로운 결과 생성용
        - *step_args: step_function 에 그대로 전달되는 실제 인자 (passage/sentences/results 등)
    """
    # 1. 캐시 우선 — 지문 안 바뀌었으면(해시 일치) 캐시 재사용 (LLM 스킵)
    cached = await cache.load_cache_if_passage_unchanged(client, cache_key, step_name, passage_text)
    if cached is not None:
        logger.info(f"[{step_name}] 지문 변경 없음. 저장된 결과 사용.")
        return cached
    logger.info(f"{step_name} - 캐시 불일치: 지문이 변경되어 새롭게 문제를 생성합니다.")

    # 2. 생성 (최대 3회 재시도)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = await step_function(*step_args)
            await cache.save_step_result(client, cache_key, step_name, result, passage_text)
            logger.info("[%s] 생성중입니다 (시도 %d회 중...)", step_name, attempt)
            return result
        except LLMError as e:
            # 비재시도(설정/요청/rate-limit 등 retryable=False) → 폴백 전에 즉시 중단
            if not e.retryable:
                logger.error("[%s] 비재시도 LLM 오류(%s) → 즉시 중단: %s", step_name, e.code, e)
                raise
            logger.warning("[%s] 재시도 %d회 실패(%s): %s", step_name, attempt, e.code, e)
        except Exception as e:
            logger.warning("[%s] 재시도 %d회 실패: %s", step_name, attempt, e)

    # 3. 폴백 — 3회 실패 시 이전 캐시라도 사용 (해시 무시)
    stale = await cache.load_cache_result(client, cache_key, step_name)
    if stale is not None:
        logger.warning("[%s] 재시도 %d회 실패 → 저장된 결과 사용", step_name, MAX_RETRIES)
        return stale

    logger.error("[%s] 재시도 %d회 실패 + 저장된 결과 없음 → 생성 실패", step_name, MAX_RETRIES)
    raise GenerationError(f"step '{step_name}' 생성 실패 (재시도/폴백 모두 실패)")


async def generate_workbook(
    client: AsyncClient | None,
    passage_text: str,
    meta: dict,
    cache_key: str,
    prompts: dict,
    grammar_addendum: str = "",
) -> str:
    """step1~8 을 의존 순서대로 조율 → 답안지 조립 → 템플릿 렌더 → 최종 HTML 문자열 반환.

    파도0(단계0): passage 만 필요 → step1,3,4,5,6 동시
    파도1(단계1): step1 산출(sentences) 필요 → step2,7 동시
    파도2(단계2): 전체 결과로 답안지 조립 + 병합/렌더

    prompts: {step_key: prompt_template} — service 에서 DB 로드해 주입.
    levels: meta['levels'] (None=전체 레벨 출력).
    """
    results: dict = {}
    
    # ── 파도0: passage 만 필요 (동시) ──
    logger.info("[generate_workbook] 단계 0 시작: step1,3,4,5,6")
    wave0 = await asyncio.gather(
        run_step(client, steps.step1_basic_analysis, cache_key, "step1_basic", passage_text,
                passage_text, prompts["step1_basic"],
                meta["user_translations"], meta["full_translation"]),
        run_step(client, steps.step3_blank, cache_key, "step3_blank", passage_text,
                passage_text, prompts["step3_blank"]),
        run_step(client, steps.step4_topic, cache_key, "step4_topic", passage_text,
                passage_text, prompts["step4_topic"]),
        run_step(client, steps.step5_grammar, cache_key, "step5_grammar", passage_text,
                passage_text, prompts["step5_grammar"], grammar_addendum),
        run_step(client, steps.step6_vocab_content, cache_key, "step6_vocab_content", passage_text,
                passage_text, prompts["step6_vocab_content"]),
    )
    results["step1"], results["step3"], results["step4"], results["step5"], results["step6"] = wave0

    # ── 파도1: step1 산출물(sentences) 필요 (동시) ──
    sentences = results["step1"].get("sentences", [])
    sentence_translations = results["step1"].get("sentence_translations", [])
    logger.info("[generate_workbook] 단계 1 시작: step2,7")
    wave1 = await asyncio.gather(
        run_step(client, steps.step2_order, cache_key, "step2_order", passage_text,
                passage_text, prompts["step2_order"], sentences),
        run_step(client, steps.step7_writing, cache_key, "step7_writing", passage_text,
                sentences, sentence_translations),
    )
    results["step2"], results["step7"] = wave1

    # ── 파도2: 전체 결과로 답안지 조립 (캐시 없이 항상 재생성 — 직접 호출) ──
    logger.info("[generate_workbook] 단계 2 시작(마지막): 답안지 조립")
    results["step8"] = answer_sheet.build_answer_sheet(results)

    # ── 병합 + 렌더 → 최종 HTML ──
    logger.info("[generate_workbook] 병합/렌더 시작")
    template_data = render.merge_to_template_data(passage_text, meta, results)
    html = render.render_workbook_html(template_data, meta.get("levels"))
    return html