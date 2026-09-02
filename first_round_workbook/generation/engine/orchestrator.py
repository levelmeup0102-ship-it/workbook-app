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
from utils.text import split_sentences, merge_short_dialogue
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
    """선택된 levels에 필요한 step만 생성 → 답안지 조립 → 렌더 → 최종 HTML 반환.

    - 지문 가공(문장 분리 + 대화문 병합)은 여기서 1회만 수행해 모든 step이 공유한다.
    - step1 산출물(sentences/translations)에 대한 의존이 사라져 파도 구분 없이 한 번에 실행.
    - levels 로 필요한 step 집합만 골라 실행 → 미선택 레벨의 LLM 호출을 생략.

    prompts: {step_key: prompt_template} — service 에서 DB 로드해 주입.
    levels: meta['levels'] (None=전체 레벨 출력).
    """
    # 1. 지문 1회 가공 (전 step 공유) — 문장 분리 후 대화문 병합
    sentences = merge_short_dialogue(split_sentences(passage_text))
    translations = meta["user_translations"]

    # 2. levels → 필요한 step 집합 결정
    #    (level 1/2/3 → 어휘·해석·문장분석은 모두 step1 산출물 사용)
    LEVEL_TO_STEP = {
        1: "step1", 2: "step1", 3: "step1",
        4: "step4", 5: "step2", 6: "step3",
        7: "step5", 8: "step6", 9: "step6", 10: "step7",
    }
    levels = meta.get("levels")
    if levels is None:
        needed_steps = {"step1", "step2", "step3", "step4", "step5", "step6", "step7"}
    else:
        needed_steps = {LEVEL_TO_STEP[level] for level in levels if level in LEVEL_TO_STEP}

    # 3. step별 실행(run_step) 조립 — 필요한 것만 lambda 로 지연 생성해 gather.
    #    가공 산출물 주입: step1/2/5 ← sentences, step7 ← sentences+translations.
    #    step3/4/6 은 passage 원문만 사용.
    step_calls = {
        "step1": lambda: run_step(client, steps.step1_basic_analysis, cache_key, "step1_basic",
                    passage_text, passage_text, sentences, prompts["step1_basic"], translations, meta["full_translation"]),
        "step2": lambda: run_step(client, steps.step2_order, cache_key, "step2_order",
                    passage_text, passage_text, prompts["step2_order"], sentences),
        "step3": lambda: run_step(client, steps.step3_blank, cache_key, "step3_blank",
                    passage_text, passage_text, prompts["step3_blank"]),
        "step4": lambda: run_step(client, steps.step4_topic, cache_key, "step4_topic",
                    passage_text, passage_text, prompts["step4_topic"]),
        "step5": lambda: run_step(client, steps.step5_grammar, cache_key, "step5_grammar",
                    passage_text, passage_text, prompts["step5_grammar"], sentences, grammar_addendum),
        "step6": lambda: run_step(client, steps.step6_vocab_content, cache_key, "step6_vocab_content",
                    passage_text, passage_text, prompts["step6_vocab_content"]),
        "step7": lambda: run_step(client, steps.step7_writing, cache_key, "step7_writing",
                    passage_text, sentences, translations),
    }
    ordered_steps = [name for name in step_calls if name in needed_steps]
    logger.info("[generate_workbook] 생성 step: %s (levels=%s)", ordered_steps, levels)
    step_outputs = await asyncio.gather(*(step_calls[name]() for name in ordered_steps))
    results: dict = dict(zip(ordered_steps, step_outputs))

    # 4. 답안지는 항상 생성하되 선택 levels 블록만 포함 + 렌더
    logger.info("[generate_workbook] 답안지 조립 + 병합/렌더")
    results["step8"] = answer_sheet.build_answer_sheet(results, levels)
    template_data = render.merge_to_template_data(passage_text, meta, results)
    html = render.render_workbook_html(template_data, levels)
    return html