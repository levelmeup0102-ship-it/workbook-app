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

logger = logging.getLogger(__name__)

TARGET_CONCURRENCY = 5

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


# 1개의 데이터에 대한 generate함수
# async def generate_one(generate_request: GenerateTarget, client: AsyncClient | None) -> dict:
#     """지문+프롬프트 로드 → /generation/engine call → 결과 반환."""
#     book, unit, pid = generate_request.book, generate_request.unit, generate_request.passage_id
#     logger.info("[generate] 요청 수신: %s/%s/%s", book, unit, pid)

#     # 1. 지문 로드
#     row, passage_text, user_translations = await _load_passage(client, book, unit, pid)

#     # 2. 캐시 키 + 메타 (템플릿/파일명용)
#     cache_key = _ck(book, unit, pid)
#     title = row.get("title", pid)
#     lesson_match = re.match(r"(\d+)", unit or "")
#     lesson_num = lesson_match.group(1) if lesson_match else "00"
#     meta = {
#         "full_translation": " ".join(user_translations), # 한글 번역: string
#         "user_translations": user_translations, # 한글 번역: list
#         "title": title,
#         "challenge_title": title,
#         "subject": book,
#         "lesson_num": lesson_num,
#         "lesson_n": lesson_num,
#         "book": book,
#         "unit": unit,
#         "levels": generate_request.levels,
#     }

#     # 3. 프롬프트 로드 + step5 어법 함정(grammar_points) 로드
#     prompts = await _load_prompts(client)
#     grammar_addendum = await get_grammar_addendum(client)

#     # 4. 엔진 조율 (파도0→1→2 + 병합/렌더) → 최종 HTML
#     logger.info("[generate] 엔진 조율 시작: cache_key=%s", cache_key)
#     html = await generate_workbook(client, passage_text, meta, cache_key, prompts, grammar_addendum)
#     logger.info("[generate] 완료: %s/%s/%s", book, unit, pid)

#     filename = f"{lesson_num}과_{title}_워크북.html"
#     return {"ok": True, "html": html, "filename": filename}

##############

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


# 준비된 데이터를 바탕으로 병렬 실행이 진행되는 함수
async def _execute_target(
    target: GenerateTarget,
    client: AsyncClient | None,
    prompts: Dict,
    grammar_addendum: str,
    semaphore: asyncio.Semaphore
) -> GenerateItemOut:
    """
    지문 1건에 대한 워크북 생성 함수를 call.
    실패시, ok=False로 처리 + 로깅
    """
    book, unit, pid = target.book, target.unit, target.passage_id
    async with semaphore:
        try:
            row, passage_text, user_translations = await _load_passage(client=client, book=book, unit=unit, pid=pid)
            cache_key = _ck(book=book, unit=unit, pid=pid)
            title = row.get("title", pid)
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
            logger.info("[generate] 완료: %s/%s/%s", book, unit, pid)
            return GenerateItemOut(ok=True, html=html, filename=filename)
        except Exception as e:
            logger.warning("[generate] 실패 %s/%s/%s: %s", book, unit, pid, e)
            return GenerateItemOut(ok=False)


# _execute_target을 call + 병렬 제어하는 함수
# 아래까지 주석 이유: GenerateItemOut에 ok나 status추가를 하지 않아서 최종 재시도 로직을 구현하지 않음.(의도적) 필요에 의해 나중에 구현될수도.
# async def run_generation_targets(
#     targets: list[GenerateTarget], client: AsyncClient | None,
#     prompts: dict, grammar_addendum: str,
# ) -> list[GenerateItemOut]:
#     """target들을 제한 병렬로 실행 → 결과 목록 반환"""
#     semaphore = asyncio.Semaphore(TARGET_CONCURRENCY)
#     results = await asyncio.gather(
#         *[_execute_target(t, client, prompts, grammar_addendum, semaphore) for t in targets]
#     )
#     return list(results)


# async def generate(payload: GenerateIn, client: AsyncClient | None) -> GenerateOut:
#     """공개 진입점: 준비 → 제한 병렬 생성 → 결과 조립."""
#     targets, prompts, grammar_addendum = await prepare_generation_data(payload, client)
#     results = await run_generation_targets(targets, client, prompts, grammar_addendum)
#     return GenerateOut(results=results)


# router -> 최종 호출 함수 -> 결과 반환 to router
async def generate(payload, client) -> GenerateOut:
    targets, prompts, grammar_addendum = await prepare_generation_data(payload, client)
    semaphore = asyncio.Semaphore(TARGET_CONCURRENCY)
    results = await asyncio.gather(
        *[_execute_target(t, client, prompts, grammar_addendum, semaphore) for t in targets]
    )
    return GenerateOut(results=list(results))