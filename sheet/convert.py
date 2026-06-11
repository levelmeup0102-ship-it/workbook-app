#!/usr/bin/env python3
"""
sheet_convert.py  ―  0회독(preclass_analysis_v24) 데이터를
                    three_modes(분석지) 프런트가 먹는 S/FLOW/TITLES/TOPICS 구조로 변환.
"""

import re
import html as _html

CIRCLED_NUMS = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮"
CIRCLED_ALPHA = "ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞ"


def _norm_num(s: str) -> str:
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
    s = (s or "").strip()
    if s and s[0] in CIRCLED_ALPHA:
        return s[0]
    if len(s) == 1 and 'a' <= s.lower() <= 'o':
        return CIRCLED_ALPHA[ord(s.lower()) - ord('a')]
    return s


_MARK_RE = re.compile(
    r'\[\[(GRAMMAR:n=([0-9]+|[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]+)(?:,split=(\d+))?'
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


_ABBR = ["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
         "e.g.", "i.e.", "etc.", "vs.", "U.S.", "U.K.", "U.S.A.",
         "Inc.", "Ltd.", "Co.", "Corp.", "No.", "Vol."]


def _split_sentences(text: str):
    prot = text or ''
    for ab in _ABBR:
        prot = prot.replace(ab, ab.replace('.', '§'))
    parts = re.split(r'(?<=[.!?])\s+(?=["\u201c\u201d]?[A-Z])', prot)
    return [p.replace('§', '.').strip() for p in parts if p.strip()]


def _split_translation(translation: str):
    if not translation:
        return []
    return [l.strip() for l in translation.splitlines() if l.strip()]


# ════════════════════════════════════════════════════════════════
# ★ VPOOL 빌더 — three_modes 가 먹는 {word: {syn:[[en,ko]], ant:[[en,ko]]}} 형태
#   (기존: {word:[syns]} 라 three_modes 의 vp.syn/vp.ant 와 안 맞았음 → 자동 안 뜸)
# ════════════════════════════════════════════════════════════════
def _pairs(raw):
    """[[en,ko],...] 정규화. 문자열만 있으면 [en, ''] 로."""
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
    """vocab_detail/vocab_notes 한 항목 → {syn, ant}."""
    return {
        "syn": _pairs(v.get("syns")),
        "ant": _pairs(v.get("ants")),
    }


def build_sheet_data(pre: dict, translation: str = "", saved_sheet: dict = None) -> dict:
    pre = pre or {}
    marked = pre.get("passage_marked", "") or ""
    original = _strip_markers(marked)

    gnotes = {}
    for n in (pre.get("grammar_notes") or []):
        if isinstance(n, dict) and n.get("num"):
            gnotes[_norm_num(n["num"])] = n
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

    tokens = _parse_marked(marked)
    sentences = _split_sentences(original)
    kr_lines = _split_translation(translation)

    chunks = []
    cursor = 0
    sent_idx = 0
    sent_bounds = _sentence_bounds(original, sentences)

    for tk in tokens:
        seg = tk["text"]
        if not seg:
            continue
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
                "h": seg.strip(), "f": 1,
                "n": _g_note(meta), "d": _g_detail(meta),
                "i": 2, "key": False, "s": "어법",
            })
        elif tk["kind"] == "V":
            lbl = tk["label"]
            meta = vnotes.get(lbl, {})
            dmeta = vdetail.get(lbl, {})
            chunk.update({
                "h": seg.strip(), "f": 2,
                "n": _v_note(meta, dmeta), "d": _v_detail(meta, dmeta),
                "i": 2, "key": False, "s": "어휘",
            })
        elif tk["kind"] == "I":
            chunk.update({"h": seg.strip(), "f": 1, "n": "함축·주제 응축",
                          "d": "빈칸/함축 출제 포인트", "i": 3, "key": True, "s": "함축"})
        chunks.append((sent_idx, chunk))

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

    # ── VPOOL (three_modes schema: {word:{syn,ant}}) ──
    # 본문 어휘(vocab_detail) + 주제어휘(theme_vocab) 둘 다 넣어
    # 클릭 시 자동으로 뜨는 단어 풀을 최대한 넓힌다.
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
    # 주제 추가 어휘
    tv = pre.get("theme_vocab") or {}
    if isinstance(tv, dict):
        for v in (tv.get("left") or []) + (tv.get("right") or []):
            if isinstance(v, dict):
                _add_vpool(v.get("word"), _vpool_entry(v))
    elif isinstance(tv, list):
        for v in tv:
            if isinstance(v, dict):
                _add_vpool(v.get("word"), _vpool_entry(v))

    POOL = {}

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


def _g_note(meta: dict) -> str:
    tag = (meta.get("tag") or "").strip()
    desc = (meta.get("desc") or "").strip()
    if tag and desc:
        return f"{tag}"
    return tag or desc or "어법 포인트"


def _g_detail(meta: dict) -> str:
    return (meta.get("desc") or "").strip()


def _v_note(meta: dict, dmeta: dict = None) -> str:
    dmeta = dmeta or {}
    syns = dmeta.get("syns") or meta.get("syns") or []
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
    return out


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
