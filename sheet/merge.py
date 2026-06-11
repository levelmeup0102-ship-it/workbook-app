#!/usr/bin/env python3
"""
sheet_merge.py  ―  강사 분석지(sheet_cache) → 0회독 자료 병합.

[흐름]
  AI 초안(step_cache.preclass_analysis_v24)  +  강사 체크(sheet_cache.sheet)
      → 병합된 passage_marked → 0회독 4페이지 자료에 반영

[병합 정책 = "나(병합)"]
  - AI 초안 마커는 유지한다.
  - 강사가 분석지에서 강조/수정한 것(sheet.marks)은 그 위에 덮어쓴다(우선).
  - 강사가 새로 추가한 단어(AI엔 없던 것)도 마커로 추가.
  - diff(난이도) / selTitle / selTopic 등 단일 선택값도 sheet 우선.

[sheet_cache.sheet jsonb shape]  (분석지 저작에서 저장)
  {
    diff: int(0~5),
    selTitle/selTopic/selSum: idx,
    tags: {si: tagText},
    marks: [
      {si, ci, h(강조구), f(1어법/2어휘), n(짧은노트), d(상세), imp(중요도1~3), key(형광bool), src}
    ],
    ...
  }

[핵심 함수]
  apply_sheet_to_preclass(pre_data, sheet) -> pre_data(병합본)
    - pre_data: generate_preclass_analysis 가 만든 dict (passage_marked 등)
    - sheet:    sheet_cache 의 sheet jsonb (없으면 None → 원본 그대로 반환)

[pipeline.py 연결]
  generate_preclass_analysis() 후처리 맨 끝, passage_html 만들기 직전에:
    sheet = _load_sheet_for_merge(cache_key)   # sheet_cache 조회 (supa 모듈)
    if sheet:
        data = apply_sheet_to_preclass(data, sheet)
    data["passage_html"] = _passage_marked_to_html(data["passage_marked"], passage)
"""
import re

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"

# 마커 파싱 정규식 (pipeline._passage_marked_to_html 과 동일 패턴)
_MARK_RE = re.compile(
    r'\[\[(GRAMMAR:n=([0-9]+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+)(?:,split=(\d+))?'
    r'|VOCAB:l=([a-oA-O]|[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ])'
    r'|IMPL)\]\]'
    r'([\s\S]*?)'
    r'\[\[/(GRAMMAR|VOCAB|IMPL)\]\]'
)


def _strip_markers(marked: str) -> str:
    return re.sub(r'\[\[/?(?:GRAMMAR|VOCAB|IMPL)[^\]]*\]\]', '', marked or '')


def _norm_num(s: str) -> str:
    s = (s or "").strip()
    if s and s[0] in CIRCLED_NUMS:
        return s[0]
    if s.isdigit() and 1 <= int(s) <= 20:
        return CIRCLED_NUMS[int(s)-1]
    return s


def _norm_alpha(s: str) -> str:
    s = (s or "").strip()
    if s and s[0] in CIRCLED_ALPHA:
        return s[0]
    if len(s) == 1 and 'a' <= s.lower() <= 'o':
        return CIRCLED_ALPHA[ord(s.lower())-ord('a')]
    return s


def _enrich_note(pre: dict, label, m: dict):
    """AI 마커에 강사 노트를 보강. label=(kind, lbl).
    강사가 더 정확한 설명을 달았으면 grammar_notes/vocab_notes 의 해당 항목 갱신.
    """
    kind, lbl = label
    note = (m.get("n") or "").strip()
    detail = (m.get("d") or "").strip()
    if kind == "G":
        for n in pre.get("grammar_notes", []) or []:
            if isinstance(n, dict) and _norm_num(n.get("num")) == lbl:
                if note:
                    n["tag"] = _short_tag(note)
                if detail:
                    n["desc"] = detail
                elif note:
                    n["desc"] = note
                break
    elif kind == "V":
        for v in pre.get("vocab_notes", []) or []:
            if isinstance(v, dict) and _norm_alpha(v.get("letter")) == lbl:
                syns = _note_to_syns(note)
                if syns:
                    v["syns"] = syns
                break


def _find_plain_in_marked(marked: str, search: str):
    """marked 에서 [[...]] 마커를 무시하고 search 위치 찾기.
    매치가 기존 마커 영역과 겹치면 다음 후보로. (pipeline._find_in_marked 축약판)
    Returns (start, end, matched_text) or None.
    """
    plain_chars, pos_map, in_marker = [], [], []
    depth, i = 0, 0
    while i < len(marked):
        if marked[i:i+2] == "[[":
            end = marked.find("]]", i)
            if end > 0:
                depth += -1 if marked[i:end+2].startswith("[[/") else 1
                depth = max(0, depth)
                i = end + 2
                continue
        plain_chars.append(marked[i]); pos_map.append(i); in_marker.append(depth > 0)
        i += 1
    plain = "".join(plain_chars)
    pat = re.compile(r'\b' + re.escape(search) + r'\b', re.IGNORECASE)
    for sm in pat.finditer(plain):
        if sm.end() > len(pos_map):
            continue
        if any(in_marker[sm.start():sm.end()]):
            continue
        s = pos_map[sm.start()]; e = pos_map[sm.end()-1] + 1
        region = marked[s:e]
        if "[[" in region or "]]" in region:
            continue
        return (s, e, region)
    return None


def _next_grammar_num(marked: str) -> str:
    """현재 passage_marked 에 안 쓰인 다음 어법 번호(원숫자)."""
    used = set(re.findall(r'\[\[GRAMMAR:n=([^,\]]+)', marked))
    used_circ = set()
    for u in used:
        u = u.strip()
        if u and u[0] in CIRCLED_NUMS:
            used_circ.add(u[0])
        elif u.isdigit() and 1 <= int(u) <= 20:
            used_circ.add(CIRCLED_NUMS[int(u)-1])
    for c in CIRCLED_NUMS:
        if c not in used_circ:
            return c
    return CIRCLED_NUMS[-1]


def _next_vocab_letter(marked: str) -> str:
    used = set(re.findall(r'\[\[VOCAB:l=([^\]]+)\]\]', marked))
    used_circ = set()
    for u in used:
        u = u.strip()
        if u and u[0] in CIRCLED_ALPHA:
            used_circ.add(u[0])
        elif len(u) == 1 and 'a' <= u.lower() <= 'o':
            used_circ.add(CIRCLED_ALPHA[ord(u.lower())-ord('a')])
    for c in CIRCLED_ALPHA:
        if c not in used_circ:
            return c
    return CIRCLED_ALPHA[-1]


def apply_sheet_to_preclass(pre: dict, sheet: dict) -> dict:
    """강사 sheet.marks 를 AI 초안 pre 에 병합 (병합 정책).

    반환: pre (in-place 수정 + 반환). passage_marked / grammar_notes /
          vocab_notes / topics 선택 등이 강사 우선으로 갱신됨.
    """
    if not isinstance(pre, dict) or not isinstance(sheet, dict):
        return pre

    marked = pre.get("passage_marked", "") or ""
    if not marked:
        return pre

    marks = sheet.get("marks") or []
    added_g, added_v, retag = 0, 0, 0

    # 강사가 강조한 각 항목을 본문에 반영
    for m in marks:
        if not isinstance(m, dict):
            continue
        h = (m.get("h") or "").strip()
        if not h:
            continue
        f = m.get("f")  # 1=어법, 2=어휘
        # 이미 그 표현이 마커로 감싸져 있나? (있으면 노트 보강, 없으면 새로 추가)
        # 강사 h 가 AI 마커 inner 와 같거나 서로 포함하면 "겹침"으로 본다.
        overlap_label = None  # (kind, label)
        for mm in _MARK_RE.finditer(marked):
            inner = (mm.group(5) or "").strip().lower()
            if not inner:
                continue
            if inner == h.lower() or inner in h.lower() or h.lower() in inner:
                if mm.group(1).startswith("GRAMMAR"):
                    overlap_label = ("G", _norm_num(mm.group(2)))
                elif mm.group(1).startswith("VOCAB"):
                    overlap_label = ("V", _norm_alpha(mm.group(4)))
                else:
                    overlap_label = ("I", None)
                break
        if overlap_label:
            # AI 가 이미 마킹함 → 마커 유지, 강사 노트로 보강 (병합 정책)
            _enrich_note(pre, overlap_label, m)
            retag += 1
            continue

        # 새 마커 추가 — 본문에서 위치 찾기
        loc = _find_plain_in_marked(marked, h)
        if not loc:
            continue
        s, e, matched = loc
        if f == 2:
            lbl = _next_vocab_letter(marked)
            wrapped = f"[[VOCAB:l={lbl}]]{matched}[[/VOCAB]]"
            marked = marked[:s] + wrapped + marked[e:]
            # vocab_notes 에도 추가 (강사 노트)
            pre.setdefault("vocab_notes", []).append({
                "letter": lbl, "word": matched,
                "syns": _note_to_syns(m.get("n")), "ants": [],
            })
            added_v += 1
        else:
            lbl = _next_grammar_num(marked)
            wrapped = f"[[GRAMMAR:n={lbl}]]{matched}[[/GRAMMAR]]"
            marked = marked[:s] + wrapped + marked[e:]
            pre.setdefault("grammar_notes", []).append({
                "num": lbl,
                "tag": _short_tag(m.get("n")),
                "desc": (m.get("d") or m.get("n") or "").strip(),
            })
            added_g += 1

    pre["passage_marked"] = marked

    # ── 단일 선택값 반영 (강사 우선) ──
    # 난이도
    if isinstance(sheet.get("diff"), int):
        pre["_teacher_diff"] = sheet["diff"]
    # 제목/주제 선택 — 강사가 고른 인덱스를 맨 앞으로 (0회독은 보통 ①을 대표로 씀)
    _reorder_by_sel(pre, "titles", sheet.get("selTitle"))
    _reorder_by_sel(pre, "titles_kr", sheet.get("selTitle"))
    _reorder_by_sel(pre, "topics", sheet.get("selTopic"))
    _reorder_by_sel(pre, "topics_kr", sheet.get("selTopic"))

    if added_g or added_v or retag:
        try:
            print(f"  [sheet-merge] 강사 강조 반영: 어법+{added_g} 어휘+{added_v} 노트보강 {retag}")
        except Exception:
            pass

    return pre


def _reorder_by_sel(pre: dict, key: str, sel):
    """sel 인덱스 항목을 리스트 맨 앞으로 (있을 때만)."""
    if not isinstance(sel, int):
        return
    arr = pre.get(key)
    if isinstance(arr, list) and 0 <= sel < len(arr):
        pre[key] = [arr[sel]] + arr[:sel] + arr[sel+1:]


def _note_to_syns(note: str):
    """강사 노트에서 동의어 추출. '≒ a · b' 처럼 동의어 마커로 시작할 때만.
    일반 설명 노트는 동의어가 아니므로 빈 리스트 반환 (오인 방지)."""
    if not note:
        return []
    note = note.strip()
    if "≒" not in note:
        return []  # 동의어 마커 없으면 동의어 아님
    note = note.split("≒", 1)[1].strip()
    parts = re.split(r'[·,/]', note)
    return [[p.strip(), ""] for p in parts if p.strip()][:3]


def _short_tag(note: str) -> str:
    """강사 노트를 짧은 태그로 (앞 12자)."""
    if not note:
        return "어법"
    note = note.strip()
    return note[:12]


# ════════════════════════════════════════════════════════════════
# pipeline 연결용 — sheet_cache 조회 (supa 모듈 재사용)
# ════════════════════════════════════════════════════════════════
def load_sheet_for_merge(cache_key: str):
    """sheet_cache 에서 강사 sheet 읽기. supa 모듈 우선, 실패 시 None.

    supa 에 sheet 전용 함수가 없으므로 REST 를 supa._request 로 직접 호출.
    (supa.py 에 SUPABASE_URL/KEY 가 이미 세팅돼 있으니 환경변수 추가 불필요)
    """
    try:
        import supa
        if not supa._enabled():
            return None
        # supa._request 는 async. pipeline 의 _run_async 로 감싸 호출.
        from urllib.parse import quote
        q = f"sheet_cache?cache_key=eq.{quote(cache_key, safe='')}&select=sheet&limit=1"
        import pipeline as pl  # _run_async 재사용
        rows = pl._run_async(supa._request("GET", q))
        if isinstance(rows, list) and rows:
            return rows[0].get("sheet")
    except Exception as e:
        try:
            print(f"  [sheet-merge] load 실패: {str(e)[:120]}")
        except Exception:
            pass
    return None


# ── CLI 검증 ──
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 3:
        print("usage: python sheet_merge.py <pre.json> <sheet.json>")
        sys.exit(1)
    pre = json.load(open(sys.argv[1], encoding="utf-8"))
    sheet = json.load(open(sys.argv[2], encoding="utf-8"))
    out = apply_sheet_to_preclass(pre, sheet)
    print("=== 병합 후 passage_marked ===")
    print(out["passage_marked"][:600])
    print("\n=== grammar_notes:", len(out.get("grammar_notes", [])))
    print("=== vocab_notes:", len(out.get("vocab_notes", [])))
