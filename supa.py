# ========================
# ★ Step Cache 일괄 조회 (로딩 속도 최적화)
# ========================
async def get_step_counts_all() -> dict:
    """step_cache의 모든 cache_key를 한 번(페이지네이션)에 받아
    {cache_key: step개수} 딕셔너리로 반환.

    ★ 목적: /api/passages 가 지문마다 count_steps()를 부르며 Supabase에
      네트워크 왕복을 수백 번 하던 것을 → 이 함수 한 번(대여섯 요청)으로 대체.
      지문 수 N에 비례하던 로딩 지연이 상수 시간으로 줄어든다.
    실패 시 빈 dict.
    """
    if not _enabled():
        return {}
    counts: dict = {}
    start = 0
    page = 0
    while True:
        end = start + PAGE_SIZE - 1
        result = await _request(
            "GET",
            "step_cache?select=cache_key&order=cache_key",
            extra_headers={"Range-Unit": "items", "Range": f"{start}-{end}"},
        )
        if not isinstance(result, list):
            break
        for r in result:
            k = r.get("cache_key")
            if k:
                counts[k] = counts.get(k, 0) + 1
        page += 1
        if len(result) < PAGE_SIZE:
            break
        start += PAGE_SIZE
        if start >= MAX_ROWS_HARD:
            print(f"[supa] get_step_counts_all: {MAX_ROWS_HARD}행 안전 상한 도달")
            break
    if page > 1:
        print(f"[supa] get_step_counts_all: {page}페이지 / cache_key {len(counts)}개")
    return counts
