#!/usr/bin/env python3
"""
sheet_convert.py  ―  0회독(preclass_analysis_v24) 데이터를
                    three_modes(분석지) 프런트가 먹는 구조로 변환.

[설계 원칙 v2 — "0회독 데이터 100% 흡수, AI 추가 호출 0"]
- 0회독은 이미 평가원 수준으로 깊게 분석돼 있다(어법 12~14박스·어휘 GRE급).
  분석지는 그걸 *다시 계산하지 않고* 그대로 빨아들여 보여주기만 한다.
- 문장 분리: pipeline.split_sentences 로직을 그대로 이식(약어·인용·이니셜 보호).
  → 자체 정규식 재분할이 일으키던 "주어 밀림" 버그 제거.
- POOL(어법 클릭): 본문 [[GRAMMAR:n=①]] 번호 ↔ grammar_p1/p2 박스 번호를 매칭해
  원리·패턴·본문예문·★유사·cf혼동을 전부 실어 보낸다.
- VPOOL(어휘 클릭): vocab_detail/vocab_notes/theme_vocab 의
  동의어 전부 + 반의어 전부 + 영영(def_en) + 예문(ex_short) 을 전부 실어 보낸다.
- 0회독에 없는 단어만 프런트가 /api/sheet/ai-synonym 으로 AI 호출(폴백).
"""

import re
import html as _html

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"


def _norm_num(s: str) -> str:
    s = (s or "").strip()
    if s and s[0] in CIRCLED_NUMS:
        return s[0]
    try:
        n = int(s)
        if 1 <= n <= 20:
            return CIRCLED_NUMS[n - 1]
    except ValueError:
        pass
    return s


def _norm_alpha(s: str) -> str:
    s = (s or "").strip()
    if s and s[0] in CIRCLED_ALPHA:
        return s[0]
    if len(s) == 1 and 'a' <= s.lower() <= 'o':
        return CIRCLED_ALPHA[ord(s.lower()) - ord('a')]
    return s


# ════════════════════════════════════════════════════════════════
# 문장 분리 — pipeline.split_sentences 이식 (약어/인용/이니셜 보호)
# ════════════════════════════════════════════════════════════════
def split_sentences(text: str) -> list:
    protected = text or ""

    def protect_quote_internals(match):
        inner = match.group(1)
        open_q = match.group(0)[0]
        close_q = match.group(0)[-1]
        protected_inner = re.sub(
            r'([.!?])\s+([A-Z])',
            lambda m2: f"{m2.group(1)}§QSEP§{m2.group(2)}",
            inner
        )
        return open_q + protected_inner + close_q

    protected = re.sub(r'["\u201c](.*?)["\u201d]', protect_quote_internals,
                       protected, flags=re.DOTALL)

    abbrevs = [
        'Dr.', 'Mr.', 'Ms.', 'Mrs.', 'Prof.', 'Jr.', 'Sr.', 'St.',
        'vs.', 'etc.', 'No.', 'Vol.', 'Fig.', 'Gen.', 'Gov.', 'Rev.',
        'Sgt.', 'Cpl.', 'Lt.', 'Co.', 'Inc.', 'Ltd.', 'Corp.', 'Dept.',
        'Est.', 'al.', 'e.g.', 'i.e.', 'U.S.', 'U.K.', 'U.N.',
    ]
    replacements = {}
    for ab in abbrevs:
        token = ab.replace('.', '§DOT§')
        pattern = r'(?<!\w)' + re.escape(ab)
        if re.search(pattern, protected):
            replacements[token] = ab
            protected = re.sub(pattern, token, protected)

    def protect_initial(m):
        return m.group(0).replace('.', '§DOT§')
    protected = re.sub(r'(?<!\w)([A-Z])\.\s*(?=[A-Z][\.\s]|[A-Z][a-z])',
                       protect_initial, protected)

    sentences = [s.strip() for s in re.split(
        r'(?<=[.!?])\s+(?=[\u201c\u201d\u0022]?[A-Z])'
        r'|(?<=[.!?][\u201c\u201d\u0022])\s+(?=[\u201c\u201d\u0022]?[A-Z])',
        protected
    ) if s.strip()]

    def inner_slice_long_quote(sentence, min_words=6):
        if '§QSEP§' not in sentence:
            return [sentence]
        parts = sentence.split('§QSEP§')

        def wc(seg):
            cleaned = seg.strip().strip('"\u201c\u201d')
            tokens = cleaned.split()
            count, i = 0, 0
            while i < len(tokens):
                t = tokens[i]
                if not any(c.isalpha() for c in t.replace('§DOT§', '')):
                    i += 1
                    continue
                while t.endswith('§DOT§') and i + 1 < len(tokens):
                    i += 1
                    t = tokens[i]
                count += 1
                i += 1
            return count

        if parts and max(wc(p) for p in parts) > min_words:
            return [p.strip() for p in parts if p.strip()]
        return [sentence]

    new_sentences = []
    for s in sentences:
        new_sentences.extend(inner_slice_long_quote(s))
    sentences = new_sentences

    restored = []
    for s in sentences:
        for token, original in replacements.items():
            s = s.replace(token, original)
        s = s.replace('§DOT§', '.')
        s = s.replace('§QSEP§', ' ')
        restored.append(s)
    return restored


# ════════════════════════════════════════════════════════════════
# passage_marked 토큰화
# ════════════════════════════════════════════════════════════════
_MARK_RE = re.compile(
    r'\[\[(GRAMMAR:n=([0-9]+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]+)(?:,split=(\d+))?'
    r'|VOCAB:l=([a-oA-O]|[ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ])'
    r'|IMPL)\]\]'
    r'([\s\S]*?)'
    r'\[\[/(GRAMMAR|VOCAB|IMPL)\]\]'
)


def _strip_markers(marked: str) -> str:
    return re.sub(r'\[\[/?(?:GRAMMAR|VOCAB|IMPL)[^\]]*\]\]', '', marked or '')


def _parse_marked(marked: str):
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
        else:
            tokens.append({"text": inner, "kind": "I",
                           "label": None, "split": False})
        last = m.end()
    if last < len(marked or ''):
        tokens.append({"text": marked[last:], "kind": None,
                       "label": None, "split": False})
    return tokens


def _split_translation(translation: str):
    if not translation:
        return []
    return [l.strip() for l in translation.splitlines() if l.strip()]


# ════════════════════════════════════════════════════════════════
# 어법 박스 인덱싱 — grammar_p1/p2 의 모든 박스를 번호별로 모음
#   박스: {num, title, lines:[{text,is_ex}]}
#   본문 [[GRAMMAR:n=①]] 의 ① ↔ 박스 num ① 로 연결
# ════════════════════════════════════════════════════════════════
def _index_grammar_boxes(pre: dict) -> dict:
    """번호(①②③..) → 박스 dict. p1/p2 의 left+right 전부."""
    boxes = {}
    for pg in ("grammar_p1", "grammar_p2"):
        g = pre.get(pg) or {}
        for side in ("left", "right"):
            for box in (g.get(side) or []):
                if not isinstance(box, dict):
                    continue
                raw = (box.get("num") or "").strip()
                # num 이 "①" 또는 드물게 "①②" 같은 병합 → 각 글자에 매핑
                for ch in raw:
                    if ch in CIRCLED_NUMS:
                        boxes.setdefault(ch, box)
    return boxes


def _box_to_pool_entry(box: dict) -> dict:
    """박스 → three_modes POOL 항목(리스트 안의 dict 하나).
    원리/패턴/본문예문/유사/cf 를 pl(제목줄)·pd(상세 멀티라인)·id(출처)로.
    """
    title = (box.get("title") or "").strip()
    lines = box.get("lines") or []
    # 화면 표시용: 각 줄을 그대로 살림 (요약/상세 토글은 프런트가 차후 결정)
    pd_parts = []
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        t = (ln.get("text") or "").strip()
        if not t:
            continue
        pd_parts.append(t)
    num = (box.get("num") or "").strip()
    return {
        "pl": title or "어법 포인트",
        "pd": "  /  ".join(pd_parts),     # 한 줄 요약용(기존 호환)
        "lines": pd_parts,                # ★ 전체 줄 (상세 표시용)
        "id": num or "어법",
    }


# ════════════════════════════════════════════════════════════════
# 어휘 VPOOL — syn 전부 + ant 전부 + 영영 + 예문
# ════════════════════════════════════════════════════════════════
def _pairs(raw):
    out = []
    for s in (raw or []):
        if isinstance(s, (list, tuple)) and len(s) >= 2:
            out.append([str(s[0]), str(s[1])])
        elif isinstance(s, (list, tuple)) and len(s) == 1:
            out.append([str(s[0]), ""])
        elif isinstance(s, str):
            out.append([s, ""])
    return out


def _vpool_entry(v: dict) -> dict:
    """vocab_detail/notes/theme 한 항목 → three_modes 가 먹는 {syn,ant,def,ex,ko}.
    동의어/반의어 전부 보존(자르지 않음). 클릭 팝업이 수준별로 다 보이게.
    """
    return {
        "syn": _pairs(v.get("syns")),
        "ant": _pairs(v.get("ants")),
        "def": (v.get("def_en") or "").strip(),
        "ex": (v.get("ex_short") or "").strip(),
        "ko": (v.get("ko") or "").strip(),
    }


# ════════════════════════════════════════════════════════════════
# 메인 변환
# ════════════════════════════════════════════════════════════════
def build_sheet_data(pre: dict, translation: str = "", saved_sheet: dict = None,
                     passage_en: str = "", wb_sentences=None, wb_translations=None) -> dict:
    pre = pre or {}
    marked = pre.get("passage_marked", "") or ""
    original = _strip_markers(marked)

    # ── 어법 노트(짧은) + 어법 박스(상세) ──
    gnotes = {}
    for n in (pre.get("grammar_notes") or []):
        if isinstance(n, dict) and n.get("num"):
            gnotes[_norm_num(n["num"])] = n
    gboxes = _index_grammar_boxes(pre)   # 번호 → 상세 박스

    # ── 어휘 메타 ──
    vnotes = {}
    for v in (pre.get("vocab_notes") or []):
        if isinstance(v, dict) and v.get("letter"):
            vnotes[_norm_alpha(v["letter"])] = v
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
    # ★ 문장 분할 = 1회독과 동일하게:
    #   1) 1회독(step1) sentences + 문장별 해석 (있으면 그대로 — 완벽)
    #   2) split_sentences(원문) 폴백 (1회독과 같은 분리기)
    #   ※ 업로드 줄바꿈은 문단 통짜로 올라온 경우 under-split 을 일으켜 쓰지 않음.
    sentences = None
    kr_lines = None
    wb_s = [str(s).strip() for s in (wb_sentences or []) if str(s).strip()]
    if len(wb_s) > 1 and _lines_match_original(wb_s, original):
        sentences = wb_s
        wb_t = [str(t).strip() for t in (wb_translations or [])]
        if len(wb_t) == len(wb_s):
            kr_lines = wb_t
    if not sentences:
        sentences = split_sentences(original)
    if kr_lines is None:
        kr_lines = _split_translation(translation)
    sent_bounds = _sentence_bounds(original, sentences)

    def _annot(tk):
        """G/V/I 토큰 → 청크에 얹을 주석 필드(없으면 None)."""
        if tk["kind"] == "G" and not tk["split"]:
            lbl = tk["label"]; note = gnotes.get(lbl, {}); box = gboxes.get(lbl, {})
            return {"h": (tk["text"] or "").strip(), "f": 1,
                    "n": _g_note(note, box), "d": _g_detail(note, box),
                    "i": 2, "key": False, "s": (note.get("num") or lbl or "어법")}
        if tk["kind"] == "V":
            lbl = tk["label"]; note = vnotes.get(lbl, {}); dmeta = vdetail.get(lbl, {})
            return {"h": (tk["text"] or "").strip(), "f": 2,
                    "n": _v_note(note, dmeta), "d": _v_detail(note, dmeta),
                    "i": 2, "key": False, "s": "어휘"}
        if tk["kind"] == "I":
            return {"h": (tk["text"] or "").strip(), "f": 1, "n": "함축·주제 응축",
                    "d": "빈칸/함축 출제 포인트", "i": 3, "key": True, "s": "함축"}
        return None

    def _sent_of(pos):
        si = 0
        for bi, (bs, _be) in enumerate(sent_bounds):
            if pos >= bs:
                si = bi
            else:
                break
        return si

    # 마커들의 원문 내 위치 + 주석 (직독직해 경로에서 구에 얹기)
    marker_spans = []   # (start, end, annot)
    _cur = 0
    for tk in tokens:
        seg = tk["text"] or ""
        st = original.find(seg, _cur) if seg else _cur
        if st < 0:
            st = _cur
        en = st + len(seg)
        _cur = en
        a = _annot(tk)
        if a:
            marker_spans.append((st, en, a))

    dt = pre.get("direct_translation")
    use_dt = (isinstance(dt, list) and len(dt) >= 2
              and all(isinstance(u, dict) and (u.get("e") or "").strip() for u in dt))

    def _split_by_sentence(ue, ust, uen):
        """[sheet][문장배치] 문장 경계를 걸친 구를 경계에서 잘라 (문장idx, 텍스트)로 반환.
        경계를 걸친 구를 통째로 시작 문장에 넣으면 다음 문장 첫 단어가 앞 블록 꼬리에 붙는다."""
        pieces, cur = [], ust
        for si, (bs, be) in enumerate(sent_bounds):
            if be <= cur or bs >= uen:
                continue
            seg = original[max(cur, bs):min(uen, be)].strip()
            if seg:
                pieces.append((si, seg))
            cur = min(uen, be)
        return pieces or [(_sent_of(ust), ue)]

    if use_dt:
        # ── 직독직해 경로: 구 단위(e,k)로 청크 구성 + 마커 얹기 (청크마다 한글) ──
        S = [[] for _ in range(max(1, len(sentences)))]
        ucur = 0
        for u in dt:
            ue = (u.get("e") or "").strip()
            uk = (u.get("k") or "").strip()
            if not ue:
                continue
            ust = original.find(ue, ucur)
            if ust < 0:
                ust = original.find(ue)
            if ust < 0:
                ust = ucur
            uen = ust + len(ue)
            ucur = uen
            for pi, (si, seg) in enumerate(_split_by_sentence(ue, ust, uen)):
                # 한글 해석은 첫 조각에만 (문장이 갈리면 뒤 조각은 비움)
                chunk = {"e": seg, "k": uk if pi == 0 else ""}
                # 이 구에 시작점이 들어오고, 마커 구가 이 조각의 영어에 실제로 포함될 때만 얹는다
                # (경계를 걸쳐 부분문자열이 아니면 렌더(split/replace)가 깨지므로 강조는 생략)
                for (ms, _me, a) in marker_spans:
                    if ust <= ms < uen and a.get("h") and a["h"] in seg:
                        chunk.update(a)
                        break
                if si >= len(S):
                    S.extend([[] for _ in range(si - len(S) + 1)])
                S[si].append(chunk)
        S = [s for s in S if s]
    else:
        # ── 폴백(토큰) 경로: 마커 토큰 단위 청크.
        #    ★ 문장 경계를 걸친 토큰(특히 마커 사이 평문)은 경계에서 잘라 각 문장에 배치.
        #      안 그러면 다음 문장의 앞부분이 앞 문장 칸에 붙어 "한 문장씩" 안 나온다.
        chunks = []  # (sent_idx, chunk)
        cursor = 0
        for tk in tokens:
            seg = tk["text"]
            if not seg:
                continue
            start = original.find(seg, cursor)
            if start < 0:
                start = cursor
            end = start + len(seg)
            cursor = end
            a = _annot(tk)

            placed = False
            for si, (bs, be) in enumerate(sent_bounds):
                if be <= start or bs >= end:
                    continue
                piece = original[max(start, bs):min(end, be)].strip()
                if not piece:
                    continue
                chunk = {"e": piece}
                # 마커(어법/어휘/함축)는 강조어가 이 조각 안에 실제로 들어갈 때만 얹기
                if a and a.get("h") and a["h"] in piece:
                    chunk.update(a)
                chunks.append((si, chunk))
                placed = True

            if not placed:
                # 안전장치: 경계 매칭 실패 시 시작 위치 문장에 통째로
                si = 0
                for bi, (bs, _be) in enumerate(sent_bounds):
                    if start >= bs:
                        si = bi
                    else:
                        break
                chunk = {"e": seg.strip()}
                if a:
                    chunk.update(a)
                chunks.append((si, chunk))

        n_sent = max([c[0] for c in chunks], default=-1) + 1
        S = [[] for _ in range(max(n_sent, len(sentences)))]
        for si, ch in chunks:
            S[si].append(ch)
        S = [s for s in S if s]
        for i, slist in enumerate(S):
            kr = kr_lines[i] if i < len(kr_lines) else ""
            for ci, ch in enumerate(slist):
                if "k" not in ch:
                    ch["k"] = kr if ci == 0 else ""

    FLOW = _build_flow(pre)
    TITLES = _zip_kr_en(pre.get("titles_kr"), pre.get("titles"))
    TOPICS = _zip_kr_en(pre.get("topics_kr"), pre.get("topics"))

    # ── VPOOL: 어휘 동의어/반의어/영영/예문 전부 ──
    VPOOL = {}

    def _add_vpool(word, entry):
        w = (word or "").strip().lower()
        if not w:
            return
        if entry["syn"] or entry["ant"]:
            VPOOL.setdefault(w, entry)

    for v in vdetail.values():
        _add_vpool(v.get("word"), _vpool_entry(v))
    for v in vnotes.values():
        _add_vpool(v.get("word"), _vpool_entry(v))
    tv = pre.get("theme_vocab") or {}
    if isinstance(tv, dict):
        for v in (tv.get("left") or []) + (tv.get("right") or []):
            if isinstance(v, dict):
                _add_vpool(v.get("word"), _vpool_entry(v))
    elif isinstance(tv, list):
        for v in tv:
            if isinstance(v, dict):
                _add_vpool(v.get("word"), _vpool_entry(v))

    # ── POOL: 어법 클릭 후보 — 본문 단어(소문자) → 박스 상세 ──
    #   본문 [[GRAMMAR:n=①]]단어 의 단어를 키로, ① 박스 상세를 값으로.
    POOL = {}
    for tk in tokens:
        if tk["kind"] != "G" or tk["split"]:
            continue
        lbl = tk["label"]
        box = gboxes.get(lbl)
        if not box:
            continue
        entry = _box_to_pool_entry(box)
        # 본문 단어 전체 + 각 단어를 키로 (three_modes 가 data-word 소문자로 찾음)
        seg = (tk["text"] or "").strip()
        if seg:
            # ★ 마킹된 구의 '원형' 키 — openAdd 가 data-an(=구 전체, 공백·대소문자 포함)으로 조회하므로 필수
            POOL.setdefault(seg, []).append(entry)
        full = re.sub(r"[^a-z']", "", seg.lower())
        if full:
            POOL.setdefault(full, []).append(entry)
        for w in re.findall(r"[A-Za-z']+", seg):
            wl = w.lower()
            if wl and wl not in POOL:
                POOL[wl] = [entry]

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

    if saved_sheet:
        out = _apply_saved(out, saved_sheet)
    return out


# ── 노트 문자열 ──
def _g_note(note: dict, box: dict) -> str:
    """한 줄 요약(저작 모드 라벨용)."""
    tag = (note.get("tag") or "").strip()
    title = (box.get("title") or "").strip()
    return tag or title or "어법 포인트"


def _g_detail(note: dict, box: dict) -> str:
    """상세 — 박스 lines 전부를 이어붙임. 없으면 note.desc."""
    lines = box.get("lines") or []
    parts = [(_l.get("text") or "").strip() for _l in lines
             if isinstance(_l, dict) and (_l.get("text") or "").strip()]
    if parts:
        return "  /  ".join(parts)
    return (note.get("desc") or "").strip()


def _v_note(note: dict, dmeta: dict) -> str:
    syns = (dmeta.get("syns") or note.get("syns") or [])
    if syns:
        chip = " · ".join(s[0] for s in syns[:3]
                          if isinstance(s, (list, tuple)) and s)
        return f"≒ {chip}" if chip else "어휘"
    return "어휘"


def _v_detail(note: dict, dmeta: dict) -> str:
    """상세 — 영영 + 동의어 전부 + 반의어 전부 + 예문."""
    parts = []
    de = (dmeta.get("def_en") or "").strip()
    if de:
        parts.append(f"[영영] {de}")
    syns = _pairs(dmeta.get("syns") or note.get("syns"))
    if syns:
        parts.append("≒ " + ", ".join(f"{s[0]}({s[1]})" for s in syns))
    ants = _pairs(dmeta.get("ants") or note.get("ants"))
    if ants:
        parts.append("↔ " + ", ".join(f"{a[0]}({a[1]})" for a in ants))
    ex = (dmeta.get("ex_short") or "").strip()
    if ex:
        parts.append(f"[예문] {ex}")
    if parts:
        return "  /  ".join(parts)
    return (dmeta.get("ko") or "").strip()


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


def _lines_match_original(lines, original) -> bool:
    """줄들이 (공백 무시) 원문과 순서대로 다 들어맞는지 확인.
    공백/개행 차이는 무시하고 대조 → 1회독 문장이 사소한 공백차로 반려되지 않게."""
    if not lines or not original:
        return False
    norm = re.sub(r"\s+", " ", original)
    cur = 0
    for ln in lines:
        nl = re.sub(r"\s+", " ", ln).strip()
        if not nl:
            continue
        idx = norm.find(nl, cur)
        if idx < 0:
            return False
        cur = idx + len(nl)
    return True


def _sentence_bounds(original: str, sentences):
    bounds = []
    cur = 0
    for s in sentences:
        idx = original.find(s, cur)
        if idx < 0:
            # 공백 차이로 실패 시: 앞 몇 단어로 근사 매칭
            probe = " ".join(s.split()[:4])
            idx = original.find(probe, cur) if probe else -1
        if idx < 0:
            idx = cur
        end = idx + len(s)
        if end > len(original):
            end = len(original)
        bounds.append((idx, end))
        cur = end
    if not bounds:
        bounds = [(0, len(original))]
    return bounds


def _apply_saved(out: dict, saved: dict) -> dict:
    if not isinstance(saved, dict):
        return out
    if isinstance(saved.get("diff"), int):
        out["DIFF"] = saved["diff"]
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
    out["_SEL"] = {
        "title": saved.get("selTitle", 0),
        "topic": saved.get("selTopic", 0),
        "sum": saved.get("selSum", 0),
        "tags": saved.get("tags", {}),
        "summary": saved.get("summary", ""),
    }
    # ★ 선생님 자유 강조(밑줄/형광/네모) 복원 — [{si,from,to,tool}]
    um = saved.get("umarks")
    if isinstance(um, list):
        out["MARKS"] = um
    elif isinstance(saved.get("boxes"), list):   # 구버전 호환
        out["MARKS"] = [{"si": b.get("si"), "from": b.get("from"),
                          "to": b.get("to"), "tool": "box"} for b in saved["boxes"]]
    return out


if __name__ == "__main__":
    import json, sys
    if len(sys.argv) < 2:
        print("usage: python convert.py <preclass_data.json> [translation.txt]")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        pre = json.load(f)
    tr = ""
    if len(sys.argv) > 2:
        with open(sys.argv[2], encoding="utf-8") as f:
            tr = f.read()
    data = build_sheet_data(pre, tr)
    print(json.dumps(data, ensure_ascii=False, indent=2))


# ════════════════════════════════════════════════════════════════
# ★ 0회독 없이 원문만으로 '빈 분석지' 만들기
#   - preclass(0회독)가 없는 지문도 분석지를 열 수 있게 한다.
#   - 문장 분리 결과로 direct_translation(dt)을 합성해 넘기면
#     build_sheet_data 가 문장별로 깔끔히 청크를 나눈다(마커/강조 없음).
#   - 선생님은 저작모드에서 단어를 눌러 직접 강조/어휘를 채운다.
# ════════════════════════════════════════════════════════════════
def build_blank_sheet_data(passage_en: str, translation: str = "", saved_sheet: dict = None,
                           wb_sentences=None, wb_translations=None) -> dict:
    raw = passage_en or ""
    passage_en = raw.strip()
    if not passage_en:
        # 원문조차 없으면 완전 빈 구조 (호출측에서 처리)
        return build_sheet_data({}, translation=translation, saved_sheet=saved_sheet)

    # 1회독 sentences 우선 → 아니면 split_sentences (업로드 문단줄은 under-split 유발해 미사용)
    wb_s = [str(s).strip() for s in (wb_sentences or []) if str(s).strip()]
    sents = wb_s if len(wb_s) > 1 else split_sentences(passage_en)

    wb_t = [str(t).strip() for t in (wb_translations or [])]
    kr_lines = wb_t if len(wb_t) == len(sents) else _split_translation(translation)
    dt = []
    for i, s in enumerate(sents):
        dt.append({"e": s, "k": (kr_lines[i] if i < len(kr_lines) else "")})

    # passage_marked 는 줄바꿈 없는 평문으로(마커 0) — 원문 그대로
    pre = {
        "passage_marked": " ".join(sents),
        "direct_translation": dt,       # 문장별 분리 유도 (len>=2일 때 dt 경로)
    }
    return build_sheet_data(pre, translation=translation, saved_sheet=saved_sheet,
                            passage_en="\n".join(sents),
                            wb_sentences=sents,
                            wb_translations=(kr_lines if len(kr_lines) == len(sents) else None))
