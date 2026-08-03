# -*- coding: utf-8 -*-
"""
A Q3 어휘 유형 (수능 30번) — 밑줄 자리 선정 코드

설계 원칙:
  · 원문(paragraphs)은 한 글자도 건드리지 않는다. 자리(인덱스)만 기록한다.
    → Q2 순서 복원 검증이 지금 코드 그대로 돌아간다. 마커 삽입·원복 왕복이 없다.
  · 밑줄 자리는 Q5 빈칸 자리를 피한다. 문자열이 아니라 인덱스 비교라 확실하다.

역할 분담 (의미 판단은 전부 LLM, 코드는 검증만):
  · 어느 단어를 밑줄 칠지    → LLM. "논지 판단에 걸리는 자리"는 의미 판정이라
    형태소 규칙(-ive, -ous)으로는 'efforts'와 'salient'를 못 가른다.
  · 동의어·반의어를 뭘로 쓸지 → LLM. 문맥에 맞는 치환어 선택도 의미 판정이다.
  · 자리가 유효한지          → 코드. 첫 문장 회피, Q5 빈칸 회피, 문장당 1개,
    중복 없음, 정답 위치 ③④⑤, 패러프레이즈 적용 여부.
  · 인덱스 보정             → 코드. LLM이 단어 위치를 자주 틀린다.
  · pick_vocab_slots()     → LLM 실패 시 폴백. 형태소 규칙으로 자리만 잡는다.

★ 문장 끝 단어(구두점 포함)도 밑줄 대상이다. 기출 2025 수능 30번 ③ 'uncomfortable.',
  상황윤리 ⑤ 'abandon' 등 문장 끝 밑줄이 흔하다. 걸러내지 않는다.

기출 7세트 35개 밑줄 실측:
  · 첫 문장에 밑줄: 0/7 (첫 문장은 논지 기준점이라 건드리지 않는다)
  · 밑줄 위치 평균: 전체의 61% 지점
  · 정답 위치: ③2회 ④3회 ⑤2회 — ①② 없음 (앞쪽이 논지를 확인시키고 뒤에서 뒤집는다)
  · 밑줄 품사: 형용사 37% / 동사 37% / 명사 17% / 부사 5%
  · 정답 품사: 형용사 4 / 동사 3 — 명사·부사가 정답인 경우 없음
"""
import re
import hashlib
from typing import Optional

# 밑줄 후보에서 제외할 기능어 — 문맥 판단 대상이 아니다
_VOCAB_STOP = {
    "a", "an", "the", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "into", "onto", "and", "or", "but", "that", "which", "who", "whose", "whom",
    "as", "than", "is", "are", "was", "were", "be", "been", "being", "am",
    "this", "these", "those", "their", "her", "his", "its", "our", "your", "my",
    "not", "no", "so", "if", "when", "while", "because", "they", "we", "i",
    "he", "she", "it", "you", "have", "has", "had", "having", "do", "does", "did",
    "may", "might", "can", "could", "will", "would", "shall", "should", "must",
    "there", "here", "what", "how", "why", "all", "some", "any", "such", "more",
    "most", "other", "another", "each", "both", "one", "two", "also", "very",
    "then", "thus", "however", "instead", "rather", "even", "only", "just",
}


# 정도·방향을 가진 어미 — 반의어 치환이 가능한 형태
_GRADABLE_SUFFIX = ("able", "ible", "ive", "ous", "ful", "less", "ant", "ent",
                    "ic", "al", "ary", "ory", "ing", "ed", "izes", "ize", "ised",
                    "ises", "ify", "ifies", "ate", "ates", "ency", "ance", "ence",
                    "ity", "ment", "tion", "sion", "ness")
# 기출 정답에 실제로 쓰인 유형 — 방향·정도를 가진 어휘
_GRADABLE_HINT = {
    "significant", "reduced", "diminished", "stronger", "weaker", "lower", "higher",
    "increase", "decrease", "enhance", "undermine", "drives", "hinders", "promotes",
    "prevents", "absolute", "relative", "internal", "external", "permanent",
    "temporary", "similar", "different", "necessary", "optional", "abandon",
    "acquire", "gain", "remove", "restore", "create", "destroy", "accept", "reject",
    "modesty", "arrogance", "justifies", "insignificant", "important", "trivial",
}
# 구체 명사 — 반의어가 없어 문제가 성립하지 않는다
_CONCRETE = {
    "tasks", "task", "work", "works", "people", "person", "thing", "things",
    "time", "times", "place", "places", "part", "parts", "case", "cases",
    "example", "examples", "point", "points", "kind", "kinds", "type", "types",
    "way", "ways", "day", "days", "year", "years", "world", "life", "lives",
    "group", "groups", "number", "numbers", "form", "forms", "area", "areas",
}


def _looks_gradable(bare: str, toks: list, i: int) -> bool:
    """반의어 치환이 가능한 단어인가 — 형용사·동사 위주로 거른다."""
    if bare in _CONCRETE:
        return False
    if bare in _GRADABLE_HINT:
        return True
    if bare.endswith(_GRADABLE_SUFFIX):
        return True
    # 3인칭 단수 동사 (-s로 끝나되 복수명사 어미는 제외)
    if len(bare) >= 6 and bare.endswith("s") and not bare.endswith(("ss", "us", "is", "ies")):
        return True
    return False


def _slot_score(bare: str, toks: list, i: int) -> int:
    """밑줄 자리로서의 적합도. 높을수록 좋다."""
    sc = 0
    if bare in _GRADABLE_HINT:
        sc += 3
    if bare.endswith(("ive", "ous", "ful", "less", "able", "ible", "ant", "ent")):
        sc += 2          # 형용사 — 기출 정답 4/7
    if bare.endswith(("ize", "izes", "ify", "ifies", "ate", "ates")):
        sc += 2          # 동사 — 기출 정답 3/7
    if bare.endswith(("tion", "sion", "ment", "ness", "ity")):
        sc -= 1          # 추상명사 — 기출 정답에 없음
    if len(bare) >= 8:
        sc += 1          # 긴 어휘가 변별력이 높다
    return sc


def _sentences(text: str) -> list:
    """문장 분리 — (시작 토큰 인덱스, 끝 토큰 인덱스, 문장) 리스트

    ★ generator.split_sentences()를 그대로 쓴다. 약어(Dr. e.g. U.S.), 1글자 이니셜,
      인용문 내부 문장까지 처리하는 검증된 분리기이고, Q2 순서 문제도 이걸로 원문을
      나눈다. 같은 지문을 Q2와 Q3가 같은 기준으로 나누게 되어 일관성이 생긴다.
      (직접 만든 규칙은 '3.5%'나 'e.g.,'를 문장 끝으로 오인해 문장 수를 틀리게 셌다.)

    반환은 토큰 인덱스라, 분리된 문장을 토큰 수로 되짚어 위치를 매긴다."""
    try:
        from variation.generator import split_sentences as _ss
        sents = _ss(text)
        # ★ split_sentences는 인용문 안 문장을 통째로 묶는다(원문 무손실이 목적이라 옳다).
        #   밑줄은 그 안에도 들어가야 하므로, 여기서만 한 겹 더 쪼갠다.
        #   따옴표가 든 문장만 한 겹 더 쪼갠다. 약어(Dr. U.S.)는 split_sentences가 이미
        #   보호했으므로, 전체를 다시 쪼개면 그 보호가 무력화된다.
        _ABBR = re.compile(r"\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|Inc|Ltd|Co|No|Vol|Fig)\.$|"
                           r"\b(?:[A-Za-z]\.){2,}$|\b[A-Z]\.$")
        deeper = []
        for sn in sents:
            if '"' in sn or "\u201c" in sn:
                parts, buf = [], ""
                for chunk in re.split(r'(?<=[.!?])\s+', sn):
                    buf = (buf + " " + chunk).strip() if buf else chunk
                    if _ABBR.search(buf):      # 약어로 끝나면 아직 문장이 안 끝났다
                        continue
                    parts.append(buf); buf = ""
                if buf:
                    parts.append(buf)
                deeper += [x for x in parts if x.strip()]
            else:
                deeper.append(sn)
        sents = deeper or sents
    except Exception:
        # generator를 못 불러오는 환경(단위 테스트 등) — 최소 규칙으로 대체
        sents = [x for x in re.split(r'(?<=[.!?])\s+(?=[A-Z"\u201c])', text) if x.strip()]

    out, cur = [], 0
    toks = text.split()
    for sent in sents:
        n = len(sent.split())
        if n == 0:
            continue
        lo, hi = cur, min(cur + n - 1, len(toks) - 1)
        if lo > hi or lo >= len(toks):
            break
        out.append((lo, hi, sent))
        cur = hi + 1
    if not out and toks:
        out = [(0, len(toks) - 1, text)]
    return out


def vocab_candidates(paragraphs, blank_spans=None, min_sent_gap=1) -> list:
    """밑줄 후보 목록. 원문은 안 건드리고 자리만 반환한다.

    paragraphs : [[label, text], ...]  — 원문 그대로
    blank_spans: {para_idx: (start_tok, end_tok)}  — Q5 빈칸이 차지한 토큰 범위
    반환       : [{"para": p, "idx": i, "word": w, "sent": s, "ratio": r}, ...]
    """
    blank_spans = blank_spans or {}
    all_sents = sum(len(_sentences(p[1])) for p in paragraphs)
    out = []
    sent_no = 0
    for p_i, (_lab, text) in enumerate(paragraphs):
        toks = text.split()
        b_lo, b_hi = blank_spans.get(p_i, (-1, -1))
        for s_lo, s_hi, _s in _sentences(text):
            sent_no += 1
            # 기출 0/7 — 지문 첫 문장에는 밑줄을 넣지 않는다
            if p_i == 0 and sent_no == 1:
                continue
            for i in range(s_lo, s_hi + 1):
                if b_lo <= i <= b_hi:            # Q5 빈칸 자리 회피
                    continue
                bare = re.sub(r"[^A-Za-z-]", "", toks[i]).lower()
                if not bare or len(bare) < 5:
                    continue
                if bare in _VOCAB_STOP or bare in _DISCOURSE_MARKER:
                    continue
                if toks[i][:1].isupper() and i != s_lo:   # 고유명사 회피
                    continue
                # ★ 반의어를 만들 수 있는 단어만 — 기출 정답은 전부 형용사 아니면 동사다.
                #   'tasks' 'time' 같은 구체명사는 반대말이 없어 문제가 성립하지 않는다.
                #   'maybe' 'often' 같은 부사도 기출 정답에 없다.
                if not _looks_gradable(bare, toks, i):
                    continue
                out.append({
                    "para": p_i, "idx": i, "word": toks[i], "bare": bare,
                    "sent": sent_no, "ratio": sent_no / max(all_sents, 1),
                    "score": _slot_score(bare, toks, i),
                })
    return out


def pick_vocab_slots(paragraphs, blank_spans=None, n=5) -> Optional[list]:
    """밑줄 5자리를 고른다. 기출 분포를 따른다.

    · 문장당 최대 1개 (한 문장에 몰리지 않게)
    · 지문 전체에 흩어지되 후반부에 무게 (기출 평균 61% 지점)
    · 정답은 ③④⑤ 자리에서 고르므로, 앞 두 자리는 전반부에 둔다
    반환: [{"n":1..5, "para":p, "idx":i, "original":w}, ...] (지문 순서)
    """
    cands = vocab_candidates(paragraphs, blank_spans)
    if len(cands) < n:
        return None

    by_sent = {}
    for c in cands:
        by_sent.setdefault(c["sent"], []).append(c)
    sents = sorted(by_sent)

    # ★ 후보가 있는 문장이 5개 미만이면, 문장당 1개 원칙을 완화해 5자리를 채운다.
    #   여기서 None을 반환하면 폴백이 통째로 실패해 항목이 어휘 없이 나간다.
    if len(sents) < n:
        picked, used = [], set()
        for s_no in sents:                       # 먼저 문장마다 하나씩
            pool = [c for c in by_sent[s_no] if c["bare"] not in used]
            if pool:
                best = max(pool, key=lambda c: c.get("score", 0))
                used.add(best["bare"]); picked.append(best)
        for c in sorted(cands, key=lambda x: -x.get("score", 0)):   # 모자란 만큼 더
            if len(picked) >= n:
                break
            if c["bare"] not in used:
                used.add(c["bare"]); picked.append(c)
        if len(picked) < n:
            return None
        picked.sort(key=lambda c: (c["para"], c["idx"]))
        return [{"n": i + 1, "para": c["para"], "idx": c["idx"], "original": c["word"]}
                for i, c in enumerate(picked[:n])]

    # 문장을 고르게 뽑되 뒤쪽에 무게 — 기출은 밑줄이 61% 지점에 몰린다
    chosen_sents = []
    total = len(sents)
    for k in range(n):
        pos = 0.15 + 0.85 * (k / max(n - 1, 1))      # 0.15 ~ 1.00
        want = sents[min(int(pos * (total - 1)), total - 1)]
        while want in chosen_sents:
            want = sents[min(sents.index(want) + 1, total - 1)]
            if want in chosen_sents and want == sents[-1]:
                break
        if want not in chosen_sents:
            chosen_sents.append(want)
    if len(chosen_sents) < n:
        for s in sents:
            if s not in chosen_sents:
                chosen_sents.append(s)
            if len(chosen_sents) == n:
                break
    chosen_sents = sorted(chosen_sents)[:n]

    slots = []
    used = set()          # ★ 같은 단어를 두 번 밑줄 치지 않는다 (기출은 전부 다른 단어)
    for s in chosen_sents:
        pool = [c for c in by_sent[s] if c["bare"] not in used]
        if not pool:
            pool = by_sent[s]          # 대안이 없으면 중복이라도 채운다
        mid = (pool[0]["idx"] + pool[-1]["idx"]) / 2
        pick = max(pool, key=lambda c: (c.get("score", 0), -abs(c["idx"] - mid)))
        used.add(pick["bare"])
        slots.append(pick)

    slots.sort(key=lambda c: (c["para"], c["idx"]))
    return [{"n": i + 1, "para": c["para"], "idx": c["idx"], "original": c["word"]}
            for i, c in enumerate(slots)]


def apply_vocab_items(paragraphs, vocab_items) -> list:
    """렌더링용 — 원문 사본에 ①~⑤ 밑줄 단어를 끼워 넣는다.
    원본 paragraphs는 그대로 두고 새 리스트를 반환한다."""
    marks = "①②③④⑤"
    out = [list(p) for p in paragraphs]
    by_para = {}
    for it in vocab_items:
        by_para.setdefault(it["para"], []).append(it)
    for p_i, items in by_para.items():
        toks = out[p_i][1].split()
        for it in items:
            i = it["idx"]
            if 0 <= i < len(toks):
                shown = it.get("shown") or it["original"]
                toks[i] = (f'<span class="vmark">{marks[it["n"] - 1]}</span>'
                           f'<u class="vword">{shown}</u>')
        out[p_i][1] = " ".join(toks)
    return out


def validate_vocab(vocab_items, paragraphs, pid="?") -> list:
    """어휘 문제 검증. 원문 대조는 별도(원문을 안 건드리므로 불필요)."""
    errors = []
    if not isinstance(vocab_items, list) or len(vocab_items) != 5:
        errors.append(f"[{pid}] [CRITICAL] Q3 어휘 밑줄은 5개여야 함 "
                      f"({len(vocab_items) if isinstance(vocab_items, list) else '형식오류'})")
        return errors

    ans = [it for it in vocab_items if it.get("is_answer")]
    if len(ans) != 1:
        errors.append(f"[{pid}] [CRITICAL] Q3 어휘 정답은 정확히 1개여야 함 ({len(ans)}개)")
    elif ans[0]["n"] < 3:
        # 기출 7세트 정답 위치: ③2 ④3 ⑤2 — ①② 없음
        errors.append(f"[{pid}] Q3 어휘 정답이 {ans[0]['n']}번 — 기출은 ③④⑤에서만 나온다. "
                      f"앞쪽 밑줄이 논지를 확인시키고 뒤에서 뒤집는 구조")

    # 자리 검증: 원문의 그 인덱스에 original이 실제로 있는가
    for it in vocab_items:
        p, i = it.get("para"), it.get("idx")
        if not isinstance(p, int) or not isinstance(i, int) or p >= len(paragraphs):
            errors.append(f"[{pid}] [CRITICAL] Q3 어휘 {it.get('n')}번 자리 정보 오류")
            continue
        toks = paragraphs[p][1].split()
        if i >= len(toks) or toks[i] != it.get("original"):
            errors.append(
                f"[{pid}] [CRITICAL] Q3 어휘 {it['n']}번 자리 불일치 — "
                f"원문[{p}][{i}]='{toks[i] if i < len(toks) else 'IDX초과'}' "
                f"vs original='{it.get('original')}'")

    # 한 문장에 두 개 이상 몰리지 않았는가
    sents = {}
    for it in vocab_items:
        p = it.get("para")
        if not isinstance(p, int) or p >= len(paragraphs):
            continue
        for s_lo, s_hi, _ in _sentences(paragraphs[p][1]):
            if s_lo <= it.get("idx", -1) <= s_hi:
                key = (p, s_lo)
                sents.setdefault(key, []).append(it["n"])
                break
    #   ★ '문장당 1개'는 권장 사항이라 오류로 올리지 않는다. generator는 오류가 하나라도
    #     있으면 재시도하므로, 경고만으로도 3회 재시도 끝에 관대 fallback으로 떨어진다.
    #     짧은 지문은 문장 수가 모자라 어쩔 수 없이 몰리기도 한다.
    crowded = [ns for ns in sents.values() if len(ns) > 1]
    if len(crowded) >= 3:      # 세 문장 이상에서 몰리면 설계가 잘못된 것
        errors.append(f"[{pid}] Q3 어휘 밑줄이 여러 문장에 몰림 {crowded} — 지문 전체에 흩을 것")

    # 문두 접속부사·담화표지는 어휘 문제로 부적절 (기출 정답 품사: 형용사4·동사3, 부사 0)
    for it in vocab_items:
        o = str(it.get("original", ""))
        if o and is_discourse_marker(o):
            kind = "정답" if it.get("is_answer") else "오답"
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번({kind}) '{o}'는 접속부사·담화표지 — "
                f"논리 흐름 표지라 문맥 판단 대상이 아니다. 형용사·동사로 고를 것")

    # 철자만 비슷한 단어로 바꿔치기 — 독해가 아니라 철자 암기를 묻게 된다
    for it in vocab_items:
        o, sh = str(it.get("original", "")), str(it.get("shown", ""))
        if o and sh and _looks_alike(o, sh):
            kind = "정답" if it.get("is_answer") else "오답"
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번({kind}) '{o}'→'{sh}'는 철자만 비슷함 — "
                f"발음·철자 유사어 금지. 뜻이 반대(정답)이거나 같은(오답) 단어로 쓸 것")

    # 같은 단어가 두 번 밑줄 — 기출은 5개가 전부 다른 단어다
    bares = [re.sub(r"[^A-Za-z-]", "", str(it.get("original", ""))).lower()
             for it in vocab_items]
    dup = [w for w in set(bares) if bares.count(w) > 1 and w]
    if dup:
        errors.append(f"[{pid}] Q3 어휘 밑줄에 같은 단어 반복 {dup} — 5개는 서로 다른 단어여야 함")

    # shown이 original과 같으면 패러프레이즈가 안 된 것
    #   ★ 폴백(코드가 자리만 잡은 것)은 shown=original이 정상이다. 면제하지 않으면
    #     폴백이 절대 통과할 수 없어 3회 재시도 끝에 항목이 통째로 fallback으로 떨어진다.
    #   ★ shown == original 이면 그 자리는 아무것도 바뀌지 않은 것이다.
    #     특히 정답 자리가 그러면 '틀린 단어'가 없어 문항이 성립하지 않는다.
    same = [it["n"] for it in vocab_items
            if str(it.get("shown", "")).strip().lower() == str(it.get("original", "")).strip().lower()]
    if same:
        _ans_same = any(it.get("is_answer") for it in vocab_items if it.get("n") in same)
        errors.append(f"[{pid}] {'[CRITICAL] ' if _ans_same else ''}"
                      f"Q3 어휘 {same}번이 원문 단어 그대로 — 바뀐 게 없어 문항이 성립하지 않는다"
                      + (" (정답 자리라 치명적)" if _ans_same else ""))

    return errors


def blank_token_spans(paragraphs) -> dict:
    """Q5 빈칸 마커(<BLANK_A>/<BLANK_B>)가 차지한 토큰 위치를 찾는다.

    마킹된 paragraphs에서 마커는 토큰 1개다. 밑줄이 그 자리를 덮으면
    빈칸 안에 밑줄이 들어가 문항이 깨지므로, 앞뒤 1토큰까지 여유를 둔다."""
    spans = {}
    for p_i, (_lab, text) in enumerate(paragraphs):
        toks = text.split()
        hits = [i for i, w in enumerate(toks)
                if "<BLANK_A>" in w or "<BLANK_B>" in w]
        if hits:
            spans[p_i] = (min(hits) - 1, max(hits) + 1)
    return spans


# 문두 접속부사·담화표지 — 기출 정답 품사는 형용사4·동사3, 부사 0.
#   'Similarly,' 'Conversely,' 같은 연결어는 문맥 판단이 아니라 논리 흐름 표지라
#   어휘 문제로 부적절하다.
_DISCOURSE_MARKER = {
    "however", "moreover", "furthermore", "therefore", "thus", "hence",
    "similarly", "conversely", "likewise", "nevertheless", "nonetheless",
    "instead", "meanwhile", "consequently", "accordingly", "besides",
    "otherwise", "indeed", "namely", "specifically", "additionally",
    "alternatively", "subsequently", "finally", "firstly", "secondly",
    "also", "then", "still", "yet", "so", "though", "although",
    "perhaps", "maybe", "probably", "certainly", "obviously", "clearly",
}


def strip_edge_punct(w: str) -> str:
    """단어 양끝 구두점 제거. 선지에 'outset:' 'discernible.' 처럼 찍히는 것을 막는다.

    본문에서는 구두점이 붙은 채로 밑줄을 쳐야 문장이 온전하지만(기출도 그렇다),
    선지 목록에는 단어만 나와야 한다."""
    return str(w or "").strip().strip('.,;:!?"\'“”‘’()[]')


def is_discourse_marker(w: str) -> bool:
    """문두 접속부사·담화표지인가."""
    return strip_edge_punct(w).lower() in _DISCOURSE_MARKER


def _looks_alike(a: str, b: str) -> bool:
    """두 단어가 '철자만 비슷한' 관계인가.

    1회독 교재도 '발음 유사 단어 절대 금지, 반드시 반의어로'를 규칙으로 둔다.
    affect/effect, adapt/adopt 같은 쌍은 독해가 아니라 철자 암기를 묻는다.
    어간이 같은 굴절형(increase/increases)은 정상이므로 제외한다."""
    import difflib
    x = re.sub(r"[^a-z]", "", str(a or "").lower())
    y = re.sub(r"[^a-z]", "", str(b or "").lower())
    if not x or not y or x == y:
        return False
    # 굴절형은 오탐 — 한쪽이 다른쪽으로 시작하면 같은 어간으로 본다
    if x.startswith(y) or y.startswith(x):
        return False
    if abs(len(x) - len(y)) > 3:
        return False
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.75


def normalize_llm_vocab(raw_items, paragraphs, blank_spans=None) -> Optional[list]:
    """LLM이 준 vocab_items를 검증·보정한다. 못 쓰면 None.

    LLM은 단어 인덱스를 자주 한두 칸 틀린다. original 문자열이 그 근처에 있으면
    실제 위치로 보정한다. 아예 못 찾으면 그 항목은 버린다."""
    if not isinstance(raw_items, list) or len(raw_items) != 5:
        return None
    out = []
    for it in raw_items:
        if not isinstance(it, dict):
            return None
        p, i = it.get("para"), it.get("idx")
        orig = str(it.get("original", "")).strip()
        if not isinstance(p, int) or p < 0 or p >= len(paragraphs) or not orig:
            return None
        toks = paragraphs[p][1].split()
        if not isinstance(i, int) or i < 0 or i >= len(toks) or toks[i] != orig:
            # ±6칸 안에서 같은 단어를 찾아 보정
            found = None
            base = i if isinstance(i, int) else 0
            for d in range(0, 7):
                for j in (base - d, base + d):
                    if 0 <= j < len(toks) and toks[j] == orig:
                        found = j
                        break
                if found is not None:
                    break
            if found is None:                     # 단락 전체에서 유일하면 그 자리로
                hits = [j for j, w in enumerate(toks) if w == orig]
                if len(hits) == 1:
                    found = hits[0]
            if found is None:
                return None
            i = found
        # Q5 빈칸 자리를 덮으면 문항이 깨진다
        if blank_spans:
            b_lo, b_hi = blank_spans.get(p, (-1, -1))
            if b_lo <= i <= b_hi:
                return None
        # 마커 자체를 밑줄로 잡은 경우
        if "<BLANK" in orig:
            return None
        _shown = str(it.get("shown", "")).strip() or orig
        out.append({
            "n": it.get("n"), "para": p, "idx": i, "original": orig,
            "shown": _shown,
            # 선지 표시용 — 구두점 뗀 형태 (본문에는 구두점 붙은 shown 을 쓴다)
            "shown_clean": strip_edge_punct(_shown),
            "original_clean": strip_edge_punct(orig),
            "is_answer": bool(it.get("is_answer")),
            "evidence_type": it.get("evidence_type", ""),
            "evidence": it.get("evidence", ""),
            "why": it.get("why", ""),
        })
    out.sort(key=lambda x: (x["para"], x["idx"]))
    for k, it in enumerate(out, 1):               # 지문 순서대로 번호 재부여
        it["n"] = k
    return out


def shuffle_answer_position(vocab_items, pid: str = "") -> list:
    """★ 폐기됨 — 아무것도 하지 않는다. 호출부 호환을 위해 남겨둔다.

    정답 위치를 코드로 옮기려 했으나 불가능하다:
      · original/shown 을 통째로 교환 → 원문[para][idx]와 original 이 어긋나
        validate_vocab 이 '자리 불일치'로 거부한다 (실측 CRITICAL 2건, 02번 A 누락 원인).
      · is_answer 만 옮기면 → 반의어가 박힌 자리가 오답이 되고 동의어 자리가 정답이 되어
        문항이 통째로 깨진다.
      · 제대로 옮기려면 새 반의어·동의어를 만들어야 하는데 그건 의미 판단이라 코드 밖이다.
    → 정답 위치 분산은 프롬프트로 유도한다 (build_vocab_prompt STEP 3)."""
    return vocab_items


def build_vocab_fallback(paragraphs, blank_spans=None) -> Optional[list]:
    """★ 폐기됨 — 항상 None 을 돌려준다.

    폴백은 자리만 잡을 뿐 반의어·동의어를 만들 수 없다(의미 판정은 코드 밖의 일).
    shown = original 인 문항은 '틀린 단어'가 아예 없어 문제가 성립하지 않는다.
    그런데도 내보내면 답지가 이렇게 나간다(실측):
        ④ proved → proved
        나머지 선지 ① patience → patience ② journalistic → journalistic ...
    이런 걸 배포하느니 None 을 돌려주고 Q3 를 기존 핵심빈칸으로 두는 편이 낫다.
    호출부는 None 을 받으면 vocab_items 를 세팅하지 않고, 템플릿이 core_blank 로 분기한다."""
    return None
