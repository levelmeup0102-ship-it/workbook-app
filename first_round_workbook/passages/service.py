"""passages 도메인 서비스 — 지문 목록/업로드/삭제 비즈니스 로직.

원본(main.py)의 로컬 dict(db["books"]...) 순회를 DB 조회(repository)로 교체.
데이터 접근만 SDK+DI 로 바꾸고 비즈니스 규칙(파싱/정렬/캐시상태)은 원본 유지.
"""
import re

from core.exceptions import BadRequestError, NotFoundError
from database import repository as repo
from utils.cache_key import _ck

from .schemas import DeletePassageIn, UploadIn

# step 캐시가 이 개수 이상이면 워크북 생성 완료(ready)로 간주 (원본: >=8)
CACHE_READY_THRESHOLD = 8

# 교재 형식 매칭 (N강/N과/Lesson/L/Chapter/Unit/N단원/SL)
_UNIT_PATTERN = re.compile(
    r"(\d+강|\d+과|Lesson\s*\d+|L\d+|Chapter\s*\d+|Unit\s*\d+|\d+단원|SL)\s*(.*)",
    re.IGNORECASE,
)


def _first_num(s: str, default: int) -> int:
    """문자열 속 첫 숫자 반환(정렬용). 없으면 default."""
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else default


def _sort_key(p: dict):
    """unit(숫자) → 수동 order 우선 → pid(숫자) 순 정렬 (원본 로직)."""
    unit_num = _first_num(p["unit"], 0)
    if p.get("order") is not None:
        return (unit_num, 0, p["order"])
    return (unit_num, 1, _first_num(p["id"], 999))


async def list_passages(client) -> list:
    """전체 지문 조회 → 정렬 -> 각 지문의 결과가 저장되어 있는지 조회 -> return count"""
    rows = await repo.get_all_passages(client)
    step_counts = await repo.count_steps_all(client)   # 1쿼리 배치 집계 (N+1 제거)
    result = []
    for row in rows:
        book, unit, pid = row.get("book"), row.get("unit"), row.get("pid")
        ck = _ck(book, unit, pid)
        cached = step_counts.get(ck, 0) >= CACHE_READY_THRESHOLD
        result.append({
            "book": book,
            "unit": unit,
            "id": pid,                                  # 프론트에서 p.id 로 씀
            "title": row.get("title", pid),
            "passage_text": row.get("passage_text", ""),
            "order": row.get("order"),                  # 수동 순서(없으면 None)
            "cache_status": "ready" if cached else "not_ready",
        })
    result.sort(key=_sort_key)
    return result


async def upload_passages(payload: UploadIn, client) -> dict:
    """###제목### 분할 파싱 → 교재형식 매칭 → DB upsert. (원본 main.py:295-312)

    ###해석### 블록은 별도 저장하지 않고 '직전 지문의 passage_text 한 컬럼'에 이어붙임.
    """
    book = (payload.book or "").strip()
    text = payload.text or ""
    if not book:
        raise BadRequestError("book 필요")
    if not text.strip():
        raise BadRequestError("text 필요")

    parts = re.split(r"###(.+?)###", text)
    rows: list[dict] = []
    last_row: dict | None = None

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        passage = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if not passage:
            continue

        # ###해석### → 직전 지문 passage_text 에 합침 (영어+한글 한 컬럼 유지)
        if title == "해석" and last_row is not None:
            last_row["passage_text"] = last_row["passage_text"] + "\n###해석###\n" + passage
            continue

        m = _UNIT_PATTERN.match(title)
        unit_name = m.group(1).strip() if m else "etc"
        pid = m.group(2).strip() if (m and m.group(2).strip()) else title

        row = {"book": book, "unit": unit_name, "pid": pid, "title": title, "passage_text": passage}
        rows.append(row)
        last_row = row

    await repo.upsert_passages_bulk(client, rows)
    return {"ok": True, "count": len(rows)}


async def delete_passage(payload: DeletePassageIn, client) -> dict:
    """단일 지문 삭제 + 그 지문 step 캐시 정리. (원본 main.py:350-390)

    - 지문에 대한 데이터가 넘어오지 않은 경우: 프론트에 재요청 + 화면에서는 지문을 다시 선택해달라고 alert
    - 삭제 대상이 없으면 백: 200 / 프론트: 선택한 지문은 DB에 없다고 alert.
    - 있으면 캐시(step_cache)도 함께 제거.
    """
    book, unit, pid = payload.book, payload.unit, payload.pid
    if not all([book, unit, pid]):
        raise BadRequestError("book, unit, pid 필요")

    deleted = await repo.delete_passage(client, book, unit, pid)
    if not deleted:
        return {"ok": True, "deleted": False}

    # 지문 삭제 성공 → 해당 지문의 생성결과 캐시 정리 (고아 데이터 방지)
    await repo.delete_steps_by_cache_key(client, _ck(book, unit, pid))
    return {"ok": True, "deleted": True}