#!/usr/bin/env python3
"""
sheet_convert.py  ―  0회독(preclass_analysis_v24) 데이터를
                    three_modes(분석지) 프런트가 먹는 S/FLOW/TITLES/TOPICS 구조로 변환.

[설계 원칙]
- three_modes 의 HTML·CSS·폰트는 한 글자도 안 바꾼다.
- 이 모듈은 "데이터만" 만든다. 만든 데이터(JS 전역 SHEET_DATA)를
  three_modes 템플릿에 주입하면 프런트가 그대로 렌더한다.
- 0회독이 이미 step_cache(preclass_analysis_v24)에 저장해 둔 분석을 재활용한다.
  (새로 AI 호출 안 함 → 비용 0, 속도 즉시)

[입력]  preclass_analysis_v24 의 data(dict). 키:
  passage_marked, passage_html, grammar_notes, vocab_notes, vocab_detail,
  topics, topics_kr, titles, titles_kr, implication_box, theme_vocab
  (+ 원문/번역은 passages.passage_text 의 ###해석### 로 분리해서 따로 넘겨줌)

[출력]  dict:
  {
    "S":      [[chunk,...], ...],   # 문장별 청크. three_modes S 와 동일 schema
    "FLOW":   [[label, text, key], ...],
    "TITLES": [[kr, en], ...],
    "TOPICS": [[kr, en], ...],
    "DIFF":   3,                    # 지문난이도 별 (기본값. sheet_cache 에 저장된 값 우선)
    "VPOOL":  {...},                # 어휘 클릭 시 동의어 칩 풀 (vocab_detail 기반)
    "POOL":   {...},                # 어법 클릭 후보 (grammar_notes 기반)
  }

S 청크 schema (three_modes 와 1:1):
  {e:영어, k:한글, h:하이라이트구, f:1=어법/2=어휘, n:짧은노트, d:상세, i:중요도1~3, key:형광bool, s:출처}
  - 마킹 안 된 평범한 조각은 {e, k} 만.
"""

import re
import html as _html

# ── 원숫자/원알파벳 테이블 (pipeline._passage_marked_to_html 과 동일) ──
CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"


def _norm_num(s: str) -> str:
    """'1'→'①', '①'→'①'."""
    s = (s or "").strip()
    if s and s[0] in CIRCLED_NUMS:
        return s[0]
    try:
        n = int(s)
        if 1 <= n <= 15:
            return CIRCLED_NUMS[n - 1]
    except ValueError:
        pass
    return s


def _norm_alpha(s: str) -> str:
    """'a'→'ⓐ', 'ⓐ'→'ⓐ'."""
    s = (s or "").strip()
    if s and s[0] in CIRCLED_ALPHA:
        return s[0]
    if len(s) == 1 and 'a' <= s.lower() <= 'o':
        return CIRCLED_ALPHA[ord(s.lower()) - ord('a')]
    return s


# ════════════════════════════════════════════════════════════════
# 1) passage_marked → 마킹 토큰 리스트
#    [[GRAMMAR:n=①]]word[[/GRAMMAR]] / [[VOCAB:l=ⓐ]]word[[/VOCAB]] / [[IMPL]]..[[/IMPL]]
# ════════════════════════════════════════════════════════════════
_MARK_RE = re.compile(
    r'\[\[(GRAMMAR:n=([0-9]+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+)(?:,split=(\d+))?'
    r'|VOCAB:l=([a-oA-O]|[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ])'
    r'|IMPL)\]\]'
    r'([\s\S]*?)'
    r'\[\[/(GRAMMAR|VOCAB|IMPL)\]\]'
)


def _strip_markers(marked: str) -> str:
    """모든 [[...]] 마커 제거 → 순수 원문."""
    return re.sub(r'\[\[/?(?:GRAMMAR|VOCAB|IMPL)[^\]]*\]\]', '', marked or '')


def _parse_marked(marked: str):
    """passage_marked 를 (텍스트조각, 마킹메타) 순서열로 토큰화.

    반환: list of dict
      {"text": "...", "kind": None|"G"|"V"|"I", "label": "①"/"ⓐ"/None, "split": bool}
    plain 텍스트와 마킹 구간을 원래 순서대로 모두 담는다.
    """
    tokens = []
    last = 0
    for m in _MARK_RE.finditer(marked or ''):
        if m.start() > last:
            tokens.append({"text": marked[last:m.start()], "kind": None,
                           "label": None, "split": False})
        kind_raw = m.group(1)
        inner = m.group(5)
        if kind_raw.startswith("GRAMMAR"):
            tokens.append({"text": inner, "kind": "G",
                           "label": _norm_num(m.group(2)),
                           "split": bool(m.group(3))})
        elif kind_raw.startswith("VOCAB"):
            tokens.append({"text": inner, "kind": "V",
                           "label": _norm_alpha(m.group(4)), "split": False})
        else:  # IMPL
            tokens.append({"text": inner, "kind": "I",
                           "label": None, "split": False})
        last = m.end()
    if last < len(marked or ''):
        tokens.append({"text": marked[last:], "kind": None,
                       "label": None, "split": False})
    return tokens


# ════════════════════════════════════════════════════════════════
# 2) 문장 분리 — 원문을 자연 문장 단위로
#    (pipeline.split_sentences 와 호환되게 약어 보호)
# ════════════════════════════════════════════════════════════════
_ABBR = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
         "e.g.", "i.e.", "etc.", "vs.", "U.S.", "U.K.", "U.S.A.",
         "Inc.", "Ltd.", "Co.", "Corp.", "No.", "Vol."]


def _split_sentences(text: str):
    prot = text or ''
    for ab in _ABBR:
        prot = prot.replace(ab, ab.replace('.', '§'))
    parts = re.split(r'(?<=[.!?])\s+(?=["\u201c\u201d]?[A-Z])', prot)
    return [p.replace('§', '.').strip() for p in parts if p.strip()]


# ════════════════════════════════════════════════════════════════
# 3) 번역(한글) 문장 매핑
#    passages.passage_text 의 ###해석### 뒤쪽을 줄 단위로.
# ════════════════════════════════════════════════════════════════
def _split_translation(translation: str):
    if not translation:
        return []
    return [l.strip() for l in translation.splitlines() if l.strip()]


# ════════════════════════════════════════════════════════════════
# 4) 메인 변환
# ════════════════════════════════════════════════════════════════
def build_sheet_data(pre: dict, translation: str = "", saved_sheet: dict = None) -> dict:
    """0회독 data(pre) → three_modes 전역 데이터(dict).

    saved_sheet: sheet_cache 에 이미 저장된 사용자 편집본이 있으면 그걸 우선 반영.
    """
    pre = pre or {}
    marked = pre.get("passage_marked", "") or ""
    original = _strip_markers(marked)

    # ── 어법/어휘 메타 인덱싱 ──
    gnotes = {}
    for n in (pre.get("grammar_notes") or []):
        if isinstance(n, dict) and n.get("num"):
            gnotes[_norm_num(n["num"])] = n
    vnotes = {}
    for v in (pre.get("vocab_notes") or []):
        if isinstance(v, dict) and v.get("letter"):
            vnotes[_norm_alpha(v["letter"])] = v
    # vocab_detail 은 {left,right} 또는 flat
    vd_raw = pre.get("vocab_detail") or []
    if isinstance(vd_raw, dict):
        vd_list = (vd_raw.get("left") or []) + (vd_raw.get("right") or [])
    else:
        vd_list = vd_raw if isinstance(vd_raw, list) else []
    vdetail = {}
    for v in vd_list:
        if isinstance(v, dict) and v.get("letter"):
            vdetail[_norm_alpha(v["letter"])] = v

    # ── 토큰 → 문장별 청크 ──
    tokens = _parse_marked(marked)
    sentences = _split_sentences(original)
    kr_lines = _split_translation(translation)

    # 토큰을 평탄화해서 "원문 오프셋 → 청크" 로 재조립.
    # 전략: 토큰 순서대로 이으면서, 문장 경계(마침표/대문자)에서 끊어 S[] 구성.
    # 각 토큰은 하나의 청크가 됨. 마킹 토큰은 h/f/n/d 채움.
    chunks = []  # (sent_index, chunk_dict)
    cursor = 0   # original 내 진행 위치
    sent_idx = 0
    sent_bounds = _sentence_bounds(original, sentences)

    for tk in tokens:
        seg = tk["text"]
        if not seg:
            continue
        # 이 조각이 걸쳐 있는 문장 인덱스 (시작 위치 기준)
        start = original.find(seg, cursor)
        if start < 0:
            start = cursor
        cursor = start + len(seg)
        while sent_idx + 1 < len(sent_bounds) and start >= sent_bounds[sent_idx + 1][0]:
            sent_idx += 1

        chunk = {"e": seg.strip()}
        if tk["kind"] == "G" and not tk["split"]:
            lbl = tk["label"]
            meta = gnotes.get(lbl, {})
            chunk.update({
                "h": seg.strip(),
                "f": 1,
                "n": _g_note(meta),
                "d": _g_detail(meta),
                "i": 2,
                "key": False,
                "s": "어법",
            })
        elif tk["kind"] == "V":
            lbl = tk["label"]
            meta = vnotes.get(lbl, {})
            dmeta = vdetail.get(lbl, {})
            chunk.update({
                "h": seg.strip(),
                "f": 2,
                "n": _v_note(meta),
                "d": _v_detail(meta, dmeta),
                "i": 2,
                "key": False,
                "s": "어휘",
            })
        elif tk["kind"] == "I":
            # 함축 — three_modes 에선 별도 f 없음. 형광 후보로만 표시.
            chunk.update({"h": seg.strip(), "f": 1, "n": "함축·주제 응축",
                          "d": "빈칸/함축 출제 포인트", "i": 3, "key": True, "s": "함축"})
        # split=1 (어법 분할 2번째 조각) 은 마킹 없이 plain 으로 둔다.
        chunks.append((sent_idx, chunk))

    # 문장별로 묶기
    n_sent = max([c[0] for c in chunks], default=-1) + 1
    S = [[] for _ in range(max(n_sent, len(sentences)))]
    for si, ch in chunks:
        S[si].append(ch)
    # 빈 문장 제거
    S = [s for s in S if s]

    # 한글 번역 붙이기 — 문장 단위로 첫 청크에 문장 전체 번역.
    # 나머지 청크는 빈 k 로 채워 정밀인쇄물에서 'undefined' 가 안 뜨게 한다.
    for i, slist in enumerate(S):
        kr = kr_lines[i] if i < len(kr_lines) else ""
        for ci, ch in enumerate(slist):
            if "k" not in ch:
                ch["k"] = kr if ci == 0 else ""

    # ── FLOW (정밀인쇄물 우측) ──
    FLOW = _build_flow(pre)

    # ── TITLES / TOPICS ──
    TITLES = _zip_kr_en(pre.get("titles_kr"), pre.get("titles"))
    TOPICS = _zip_kr_en(pre.get("topics_kr"), pre.get("topics"))

    # ── VPOOL (어휘 동의어 칩) ──
    VPOOL = {}
    for lbl, v in vdetail.items():
        w = (v.get("word") or "").lower()
        syns = [s[0] for s in (v.get("syns") or []) if isinstance(s, (list, tuple)) and s]
        if w and syns:
            VPOOL[w] = syns
    for lbl, v in vnotes.items():
        w = (v.get("word") or "").lower()
        if w and w not in VPOOL:
            syns = [s[0] for s in (v.get("syns") or []) if isinstance(s, (list, tuple)) and s]
            if syns:
                VPOOL[w] = syns

    # ── POOL (어법 클릭 후보) ──
    POOL = {}
    for lbl, n in gnotes.items():
        # 노트의 desc 첫 단어를 키로 쓰긴 어려워 word 추정 불가 → tag 기반 일반 후보
        pass  # three_modes 의 POOL 은 단어→후보. 0회독엔 단어 매핑이 없어 비워둔다(직접작성 fallback).

    out = {
        "S": S,
        "FLOW": FLOW,
        "TITLES": TITLES or [["", ""]],
        "TOPICS": TOPICS or [["", ""]],
        "DIFF": 3,
        "VPOOL": VPOOL,
        "POOL": POOL,
        "ORIGINAL": original,
    }

    # ── 저장된 사용자 편집본 우선 반영 ──
    if saved_sheet:
        out = _apply_saved(out, saved_sheet)
    return out


# ──────────────────────────────────────────────────────────────
# 헬퍼: 노트 문자열 만들기
# ──────────────────────────────────────────────────────────────
def _g_note(meta: dict) -> str:
    tag = (meta.get("tag") or "").strip()
    desc = (meta.get("desc") or "").strip()
    if tag and desc:
        # desc 가 길면 앞부분만 (three_modes n 은 짧은 한 줄)
        return f"{tag}"
    return tag or desc or "어법 포인트"


def _g_detail(meta: dict) -> str:
    return (meta.get("desc") or "").strip()


def _v_note(meta: dict) -> str:
    syns = meta.get("syns") or []
    if syns:
        chip = " · ".join(s[0] for s in syns[:3] if isinstance(s, (list, tuple)) and s)
        return f"≒ {chip}" if chip else "어휘"
    return "어휘"


def _v_detail(meta: dict, dmeta: dict) -> str:
    ants = (dmeta.get("ants") or meta.get("ants") or [])
    if ants:
        a = ants[0]
        if isinstance(a, (list, tuple)) and len(a) >= 2:
            return f"↔ {a[0]}({a[1]})"
        if isinstance(a, (list, tuple)) and a:
            return f"↔ {a[0]}"
    ko = dmeta.get("ko") or ""
    return ko


def _build_flow(pre: dict):
    box = pre.get("implication_box") or {}
    panels = box.get("flow_4panel") or []
    flow = []
    for p in panels:
        if not isinstance(p, dict):
            continue
        stage = (p.get("stage") or "").strip()
        label_en = (p.get("label_en") or "").strip()
        label = f"&lt;{stage}&gt;" + (f" {label_en}" if label_en else "")
        text = (p.get("text_kr") or "").strip()
        key = (p.get("key_phrase") or "").strip()
        flow.append([label, text, key])
    return flow


def _zip_kr_en(kr_list, en_list):
    kr_list = kr_list or []
    en_list = en_list or []
    n = max(len(kr_list), len(en_list))
    out = []
    for i in range(n):
        kr = kr_list[i] if i < len(kr_list) else ""
        en = en_list[i] if i < len(en_list) else ""
        out.append([kr, en])
    return out


def _sentence_bounds(original: str, sentences):
    """각 문장의 (시작오프셋, 끝오프셋) 리스트."""
    bounds = []
    cur = 0
    for s in sentences:
        idx = original.find(s, cur)
        if idx < 0:
            idx = cur
        bounds.append((idx, idx + len(s)))
        cur = idx + len(s)
    if not bounds:
        bounds = [(0, len(original))]
    return bounds


# ──────────────────────────────────────────────────────────────
# 저장본(sheet_cache.sheet jsonb) 적용
#   shape: {diff, selTitle, selTopic, selSum, tags:{si:txt},
#           marks:[{si,ci,h,f,n,d,imp,key,src}], flow, summary, ...}
# ──────────────────────────────────────────────────────────────
def _apply_saved(out: dict, saved: dict) -> dict:
    if not isinstance(saved, dict):
        return out
    if isinstance(saved.get("diff"), int):
        out["DIFF"] = saved["diff"]
    # marks override: si/ci 로 청크 갱신
    for m in saved.get("marks", []) or []:
        try:
            si, ci = m.get("si"), m.get("ci")
            if si is None:
                continue
            sl = out["S"][si]
            if ci is None or ci >= len(sl):
                continue
            ch = sl[ci]
            for k_src, k_dst in (("h", "h"), ("f", "f"), ("n", "n"),
                                 ("d", "d"), ("imp", "i"), ("key", "key"), ("src", "s")):
                if k_src in m:
                    ch[k_dst] = m[k_src]
        except (IndexError, KeyError, TypeError):
            continue
    # 선택 인덱스/요약 등은 프런트가 SHEET_SEL 로 읽음 → 그대로 통과
    out["_SEL"] = {
        "title": saved.get("selTitle", 0),
        "topic": saved.get("selTopic", 0),
        "sum": saved.get("selSum", 0),
        "tags": saved.get("tags", {}),
        "summary": saved.get("summary", ""),
    }
    return out


# ──────────────────────────────────────────────────────────────
# 디버그/검증용 CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("usage: python sheet_convert.py <preclass_data.json> [translation.txt]")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        pre = json.load(f)
    tr = ""
    if len(sys.argv) > 2:
        with open(sys.argv[2], encoding="utf-8") as f:
            tr = f.read()
    data = build_sheet_data(pre, tr)
    print(json.dumps(data, ensure_ascii=False, indent=2))
