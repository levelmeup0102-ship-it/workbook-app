"""generation 도메인 서비스 — /api/generate 비즈니스 로직.

라우터(HTTP)와 엔진(생성)을 잇는 조합(composition) 계층:
DB에서 지문·프롬프트를 로드해 orchestrator 에 주입.
"""
import asyncio
import re
import logging

from supabase import AsyncClient

from database import repository
from utils.cache_key import _ck
from typing import List, Dict
from core.exceptions import NotFoundError
from llm.prompt_service import get_prompt_from_database, get_grammar_addendum

from .schemas import GenerateTarget, GenerateIn, GenerateOut, GenerateItemOut
from .engine.orchestrator import generate_workbook
from .job_manager import JobManager
from .job_models import GenerationJob

logger = logging.getLogger(__name__)

TARGET_CONCURRENCY = 4

def _split_passage_and_translation(raw: str) -> tuple[str, list]:
    """passage_text 에서 영어 지문 / 한글 번역 분리.(기준: ###해석###)"""
    if "###해석###" in raw: # 해석이 있는 경우에는,
        eng, kr = raw.split("###해석###", 1)
        translations = [line.strip() for line in kr.strip().splitlines() if line.strip()]
        return eng.strip(), translations
    return raw.strip(), [] # 해석이 없다면, 한글 번역은 빈 List를 return


async def _load_passage(client: AsyncClient | None, book: str, unit: str, pid: str) -> tuple[dict, str, list]:
    """지문 조회 + 영어/번역 분리. 없으면 NotFoundError(→ 404). (row, passage_text, translations)"""
    row = await repository.get_passage(client=client, book=book, unit=unit, pid=pid)
    if not row:
        raise NotFoundError(f"passage not found: {book}/{unit}/{pid}")
    passage_text, user_translations = _split_passage_and_translation(row.get("passage_text", ""))
    logger.info("지문 로드 완료: %s/%s/%s (번역 %d줄)", book, unit, pid, len(user_translations))
    return row, passage_text, user_translations


async def _load_prompts(client: AsyncClient | None) -> dict:
    """활성 프롬프트 dict 로드 후 개수 로깅. (미등록 로깅은 prompt_service 담당)"""
    prompts = await get_prompt_from_database(client)
    logger.info("프롬프트 %d개 로드 완료", sum(1 for tmpl in prompts.values() if tmpl))
    return prompts

# bulk Input 값 평탄화
def normalize_generate_targets(payload: GenerateIn) -> List[GenerateTarget]:
        """Input으로 받은 지문 정보를 GenerateTarget타입으로 평탄화."""
        targets: List[GenerateTarget] = []
        for unit_item in payload.units:
                for passage_id in unit_item.passage_ids:
                        targets.append(
                                GenerateTarget(
                                        book=payload.book,
                                        unit=unit_item.unit,
                                        passage_id=passage_id,
                                        levels=payload.levels
                                        )
                                )
        return targets



# 워크북 생성을 위한 데이터 준비 함수
async def prepare_generation_data(
    payload: GenerateIn,
    client: AsyncClient | None
) -> tuple[list[GenerateTarget], dict, str]:
    """
    1개의 요청에 필요한 모든 데이터 준비: 프롬프트/문법 로드
    """
    targets = normalize_generate_targets(payload)
    prompts = await _load_prompts(client)
    grammar_addendum = await get_grammar_addendum(client)

    return targets, prompts, grammar_addendum


# 지문 수만 세는 경량 계산(DB 접근 없음) — router 의 sync/job 분기용
def count_targets(payload: GenerateIn) -> int:
    """payload 의 총 지문 수 = 모든 unit 의 passage_ids 합."""
    return sum(len(unit.passage_ids) for unit in payload.units)


# job 생성 경로 — prepare 1회 후 job_manager 에 등록(router 는 prepare 를 호출하지 않음)
async def create_generation_job(
    payload: GenerateIn,
    client: AsyncClient | None,
    job_manager: JobManager,
) -> GenerationJob:
    """대량 요청을 대기열 job 으로 등록하고 생성된 job 을 반환한다."""
    targets, prompts, grammar_addendum = await prepare_generation_data(payload, client)
    return await job_manager.create_job(targets, prompts, grammar_addendum)


# 지문 1건 생성(순수) — 세마포어 없음. sync 경로와 worker 가 공용으로 사용.
async def execute_generation_target(
    *,
    target: GenerateTarget,
    client: AsyncClient | None,
    prompts: Dict,
    grammar_addendum: str,
) -> GenerateItemOut:
    """지문 1건에 대한 워크북 생성.
    성공 시 GenerateItemOut(ok=True, ...) 반환, 실패 시 예외를 그대로 올린다(호출측이 처리).
    (worker → mark_failed_or_retry / sync → _safe_execute_target 에서 처리)
    """
    book, unit, passage_id = target.book, target.unit, target.passage_id
    row, passage_text, user_translations = await _load_passage(client=client, book=book, unit=unit, pid=passage_id)
    cache_key = _ck(book=book, unit=unit, pid=passage_id)
    title = row.get("title", passage_id)
    lesson_match = re.match(r"(\d+)", unit or "")
    lesson_num = lesson_match.group(1) if lesson_match else "00"
    meta = {
        "full_translation": " ".join(user_translations),
        "user_translations": user_translations,
        "title": title, "challenge_title": title,
        "subject": book, "lesson_num": lesson_num, "lesson_n": lesson_num,
        "book": book, "unit": unit, "levels": target.levels,
    }
    html = await generate_workbook(client, passage_text, meta, cache_key, prompts, grammar_addendum)
    filename = f"{lesson_num}과_{title}_워크북.html"
    logger.info("[generate] 완료: %s/%s/%s", book, unit, passage_id)
    return GenerateItemOut(ok=True, html=html, filename=filename)


# sync 경로용 안전 래퍼 — 한 지문 실패가 전체를 죽이지 않도록 예외를 잡아 ok=False 반환.
async def _safe_execute_target(
    target: GenerateTarget,
    client: AsyncClient | None,
    prompts: Dict,
    grammar_addendum: str,
) -> GenerateItemOut:
    try:
        return await execute_generation_target(
            target=target, client=client, prompts=prompts, grammar_addendum=grammar_addendum,
        )
    except Exception as exception:
        logger.warning("[generate] 실패 %s/%s/%s: %s",
                       target.book, target.unit, target.passage_id, exception)
        return GenerateItemOut(ok=False)




# router -> sync 최종 호출 함수 -> 결과 반환 to router
# (지문 수가 적을 때만 호출됨. 많으면 router가 job 대기열로 우회)
async def generate(payload, client) -> GenerateOut:
    targets, prompts, grammar_addendum = await prepare_generation_data(payload, client)
    semaphore = asyncio.Semaphore(TARGET_CONCURRENCY)

    async def run_with_limit(target: GenerateTarget) -> GenerateItemOut:
        async with semaphore:   # 동시에 생성하는 지문 수 제한
            return await _safe_execute_target(target, client, prompts, grammar_addendum)

    results = await asyncio.gather(*[run_with_limit(target) for target in targets])
    return GenerateOut(results=list(results))