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
    ★ 단, 이 분포는 원문 순서 지문 기준이다(_s97). 우리 A 지문은 Q2 순서배열 때문에
      (A)(B)(C)가 셔플돼 학생이 보는 순서와 원문 순서가 다르다.
      '앞/뒤' 개념이 없으므로 위치 가중치를 쓰지 않는다. 고르게 흩기만 한다.
  · 정답 위치: ③2회 ④3회 ⑤2회 — 같은 이유로 제약하지 않는다. ①~⑤ 전부 쓴다(_s96).
  · 밑줄 품사: 형용사 37% / 동사 37% / 명사 17% / 부사 5%
  · 정답 품사: ★ 명사도 정답이 된다(_s97). 2015 수능 30번 정답이 'concern'(명사)이고,
    연성 하천공학 'modesty', 바자르 'restrictions' 도 명사다.
    기준은 품사가 아니라 '이 문맥에서 방향을 뒤집을 수 있는가'다.
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
# 방향이 없는 구체 명사 — 무엇으로 바꿔도 논지가 뒤집히지 않는다.
#   ★ '명사라서' 막는 게 아니다(_s97). 방향이 있는 명사는 정답이 된다 —
#     기출 'concern'(상황윤리 ⑤ 정답), 'modesty', 'restrictions', 'majority'.
#     여기 목록은 반대말 자체를 댈 수 없는 것들만 남긴다.
_CONCRETE = {
    "tasks", "task", "work", "works", "people", "person", "thing", "things",
    "time", "times", "place", "places", "part", "parts", "case", "cases",
    "example", "examples", "point", "points", "kind", "kinds", "type", "types",
    "way", "ways", "day", "days", "year", "years", "world", "life", "lives",
    "group", "groups", "number", "numbers", "form", "forms", "area", "areas",
}


# 방향을 가진 추상명사 — 기출 정답에 실제로 쓰인다(_s97).
#   'concern'(상황윤리 ⑤) 'modesty'(연성 하천공학) 'restrictions'(바자르) 'necessity'
_DIRECTIONAL_NOUN = {
    "concern", "modesty", "arrogance", "majority", "minority", "presence",
    "absence", "necessity", "restriction", "restrictions", "freedom", "constraint",
    "advantage", "disadvantage", "benefit", "harm", "gain", "loss", "surplus",
    "shortage", "excess", "scarcity", "abundance", "priority", "neglect",
    "consensus", "dispute", "certainty", "doubt", "clarity", "ambiguity",
    "stability", "volatility", "autonomy", "dependence", "openness", "secrecy",
    "trust", "suspicion", "growth", "decline", "progress", "regression",
}


def _looks_gradable(bare: str, toks: list, i: int) -> bool:
    """반의어 치환이 가능한 단어인가.

    ★ 기준은 품사가 아니라 '이 문맥에서 방향을 뒤집을 수 있는가'다(_s97).
      명사도 방향이 있으면 정답이 된다 — 2015 수능 30번 정답이 'concern' 이다.
      막아야 할 것은 방향 없는 구체명사(tasks, time, readers)뿐이다."""
    if bare in _CONCRETE:
        return False
    if bare in _GRADABLE_HINT or bare in _DIRECTIONAL_NOUN:
        return True
    if bare.endswith(_GRADABLE_SUFFIX):
        return True
    # 3인칭 단수 동사 / 복수형 (-s로 끝나되 명백한 예외는 제외)
    if len(bare) >= 6 and bare.endswith("s") and not bare.endswith(("ss", "us", "is")):
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
    if bare in _DIRECTIONAL_NOUN:
        sc += 3          # 방향 있는 추상명사 — 기출 정답에 실제로 쓰인다(concern, modesty)
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
                    # ratio 는 지문 내 상대 위치. _s97부터 자리 선정에는 쓰지 않는다
                    # (셔플돼 있어 앞뒤 개념이 없다). 진단용으로만 남긴다.
                    "sent": sent_no, "ratio": sent_no / max(all_sents, 1),
                    "score": _slot_score(bare, toks, i),
                })
    return out


def pick_vocab_slots(paragraphs, blank_spans=None, n=5) -> Optional[list]:
    """밑줄 5자리를 고른다. 기출 분포를 따른다.

    · 문장당 최대 1개 (한 문장에 몰리지 않게)
    · 지문 전체에 고르게 흩는다
      ★ 후반부 가중치를 쓰지 않는다(_s97). 기출이 61% 지점에 몰리는 것은
        원문 순서 지문이라 '앞에서 확인시키고 뒤에서 뒤집는' 구조가 성립하기 때문이다.
        우리 A 지문은 Q2 순서배열 때문에 (A)(B)(C)가 셔플돼 앞뒤 개념이 없다.
    · 정답 자리도 ①~⑤ 전부 쓴다(_s96) — 같은 이유다.
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

    # ★ 문장을 처음부터 끝까지 고르게 뽑는다 (_s97).
    #   옛 코드는 0.15~1.00 으로 뒤쪽에 무게를 뒀는데, 그건 원문 순서 지문 기준이다.
    #   우리 지문은 셔플돼 있어 '뒤쪽'이라는 게 없다. 균등 분할이 맞다.
    chosen_sents = []
    total = len(sents)
    for k in range(n):
        pos = (k + 0.5) / n                          # 0 ~ 1 균등
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
    # ★ 정답 자리 ③④⑤ 제약은 _s96에서 해제했다.
    #   기출이 ③④⑤뿐인 이유는 원문 지문 순서가 고정이라 '앞쪽 밑줄이 논지를
    #   확인시키고 뒤에서 뒤집는' 구조가 성립하기 때문이다.
    #   우리 A 지문은 Q2 순서배열 때문에 (A)(B)(C)가 셔플돼 있다. 학생이 보는 순서와
    #   원문 순서가 다르므로 '앞뒤' 개념 자체가 없다. ①~⑤ 전부 정답이 될 수 있다.

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

    # ★ 정답 자리는 형용사·동사여야 한다. 구체명사는 반의어가 없어 반전이 불가능하다.
    for it in vocab_items:
        if not it.get("is_answer"):
            continue
        o, sh = str(it.get("original", "")), str(it.get("shown", ""))
        if not answer_pos_ok(o) or not answer_pos_ok(sh):
            _bad = o if not answer_pos_ok(o) else sh
            errors.append(
                f"[{pid}] [CRITICAL] Q3 어휘 정답 자리 '{_bad}'는 반의어를 만들 수 없는 품사 — "
                f"기준은 품사가 아니라 '반대말을 댈 수 있는가'다 — "
                f"기출에도 명사 정답이 있다(concern, modesty, restrictions). "
                f"방향을 가진 말로 고를 것")

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

    # ★ 형태(굴절) 일치 — 지문이 'depends' 인데 shown 이 'rely' 면 본문 수일치가 깨진다.
    #   normalize 단계에서 이미 걸러지지만, 폴백 경로로 들어온 것도 있으므로 백스톱을 둔다.
    for it in vocab_items:
        _sm = shape_mismatch(str(it.get("original", "")), str(it.get("shown", "")))
        if _sm:
            errors.append(f"[{pid}] [CRITICAL] Q3 어휘 {it.get('n')}번 형태 불일치 — {_sm}")

    # 같은 단어가 두 번 밑줄 — 기출은 5개가 전부 다른 단어다
    #   ★ 굴절형도 같은 단어로 본다(_s97). 'depends' 와 'depend' 는 학생 눈에 같은 말이라
    #     ②③에 나란히 나오면 "왜 같은 단어가 두 번이지" 한다. 기출은 5개가 전부 다르다.
    def _stem(w):
        x = re.sub(r"[^A-Za-z-]", "", str(w or "")).lower()
        for suf in ("ies", "ied", "ing", "es", "ed", "s"):
            if x.endswith(suf) and len(x) - len(suf) >= 3:
                base = x[:-len(suf)]
                if suf in ("ies", "ied"):
                    base += "y"
                return base
        return x
    bares = [_stem(it.get("original", "")) for it in vocab_items]
    dup = [w for w in set(bares) if bares.count(w) > 1 and w]
    if dup:
        _pairs = [str(it.get("original", "")) for it in vocab_items
                  if _stem(it.get("original", "")) in dup]
        errors.append(f"[{pid}] Q3 어휘 밑줄에 같은 단어 반복 {_pairs} — "
                      f"5개는 서로 다른 단어여야 한다 (굴절형도 같은 단어로 본다)")

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


# 형용사·동사로 보이는 어미. ★ 명사도 방향이 있으면 정답이 된다(_s97) —
# 기출 'concern'(2015 수능 30번 ⑤ 정답), 'modesty', 'restrictions'.
_ADJ_SUFFIX = ("able", "ible", "ive", "ous", "ful", "less", "ant", "ent",
               "ic", "ical", "al", "ary", "ory", "ish", "like", "worthy",
               "ed", "ing", "en")
_VERB_SUFFIX = ("ize", "ise", "ify", "ate", "en", "ed", "ing", "s", "es")
# 명사임이 확실한 어미 — 이런 걸 정답 자리에 두면 반의어를 만들 수 없다
_NOUN_SUFFIX = ("tion", "sion", "ment", "ness", "ity", "ance", "ence",
                "ship", "hood", "dom", "ism", "ist", "ure", "age", "cy")
# 자주 나오는 구체명사 (어미로는 안 걸린다)
_NOUN_HARD = {
    "story", "stories", "line", "lines", "word", "words", "book", "books",
    "hand", "hands", "eye", "eyes", "face", "faces", "head", "heads",
    "house", "room", "city", "country", "school", "student", "students",
    "teacher", "child", "children", "man", "men", "woman", "women",
    "water", "food", "money", "car", "cars", "road", "roads", "door",
    "reader", "readers", "writer", "writers", "author", "authors",
    "name", "names", "idea", "ideas", "fact", "facts", "reason", "reasons",
    "result", "results", "problem", "problems", "question", "questions",
    "answer", "answers", "step", "steps", "level", "levels", "side", "sides",
}


# 어미가 없어 형태로는 판정 안 되는 원형 동사 — 실제 산출물 정답에 쓰인 것들
_VERB_BASE = {
    "rely", "depend", "disregard", "ignore", "reject", "accept", "deny",
    "allow", "prevent", "enable", "hinder", "help", "harm", "cause",
    "reduce", "raise", "lower", "boost", "curb", "limit", "expand",
    "shrink", "grow", "fade", "rise", "fall", "gain", "lose", "keep",
    "drop", "hold", "block", "clear", "open", "close", "start", "stop",
    "begin", "end", "join", "split", "unite", "divide", "share", "hide",
    "reveal", "conceal", "seek", "avoid", "meet", "miss", "win", "fail",
    "succeed", "improve", "worsen", "protect", "attack", "defend", "resist",
    "obey", "follow", "lead", "trail", "value", "waste", "save", "spend",
    "trust", "doubt", "confirm", "refute", "affirm", "oppose", "support",
    "favor", "shun", "embrace", "abandon", "retain", "discard", "claim",
    "refuse", "grant", "withhold", "attract", "repel", "ease", "strain",
}


def answer_pos_ok(word: str) -> bool:
    """정답 자리에 쓸 수 있는 단어인가.

    ★ 기준은 '품사'가 아니라 '반대말이 있는가'다.
      명사도 방향이 있으면 정답이 된다 — 기출에 실제로 쓰인다:
        modesty ↔ arrogance (연성 하천공학) / restrictions (바자르)
        concern (상황윤리) / necessity / majority ↔ minority
      쓸 수 없는 건 방향이 없는 구체명사다:
        story / tasks / readers / line — 반대말 자체가 없다
    ★ 그런데 'story' 와 'modesty' 는 형태로 구분할 수 없다(둘 다 명사).
      의미 판단이라 LLM 몫이고, 코드는 확실한 것만 거른다:
        · 대명사·기능어  (문맥 판단 대상이 아님)
        · 접속부사       (논리 흐름 표지)
        · 부사(-ly)      (기출 정답에 없음)
      반의어를 못 만든 경우는 shown == original 검사가 따로 잡는다."""
    w = re.sub(r"[^A-Za-z-]", "", str(word or "")).lower()
    if not w or len(w) < 3:
        return False
    if w in _VOCAB_STOP:                             # 관사·전치사·대명사 등 기능어
        return False
    if w in _DISCOURSE_MARKER:                       # 접속부사 — 문맥 판단 대상 아님
        return False
    # -ly 는 대개 부사. rely/apply 처럼 동사이거나 early/likely 처럼 형용사인 것만 허용
    _LY_OK = {"rely", "apply", "imply", "comply", "supply", "reply", "multiply",
              "early", "only", "likely", "costly", "friendly", "lonely",
              "lively", "timely", "ugly", "holy", "silly", "daily", "deadly"}
    if w.endswith("ly") and w not in _LY_OK:
        return False
    return True


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
    # ★ 부정 접두사로 만든 반의어는 정상이다. 철자가 비슷할 수밖에 없다.
    #   inhabitable/uninhabitable, possible/impossible, regular/irregular 등.
    #   실측: 'inhabitable'→'uninhabitable' 가 철자유사로 거부돼 3회 재시도 끝에
    #   관대 fallback 으로 떨어졌다. affect/effect 같은 혼동어와는 전혀 다르다.
    _NEG = ("un", "in", "im", "il", "ir", "dis", "non", "a", "anti", "de", "mis")
    _lo, _hi = (x, y) if len(x) <= len(y) else (y, x)
    for _p in _NEG:
        if _hi == _p + _lo:
            return False
    if abs(len(x) - len(y)) > 3:
        return False
    return difflib.SequenceMatcher(None, x, y).ratio() >= 0.75


# ════════════════════════════════════════════════════════════════
# 형태(굴절) 판정 — 코드는 '고치지 않고' 어긋났다는 사실만 알려준다 (_s97)
#   지문이 'depends'(3인칭 단수)면 치환어도 'relies'여야 한다. 'rely'를 넣으면
#   본문이 'the brain rely on ...' 로 수일치가 깨진다.
#   ★ 코드가 어미를 고쳐 통과시키면 안 된다 — 'depends' 를 보고 'rely' 를 냈다는 건
#     문맥을 제대로 안 봤다는 뜻이고, 어미만 고치면 그 부주의가 그대로 남는다.
#     어긋난 사실을 사유로 돌려주고 LLM 이 다시 만들게 한다.
# ════════════════════════════════════════════════════════════════
def _word_shape(w: str) -> str:
    """단어의 굴절 형태를 대략 판정한다. 정확한 품사 분석이 아니라
    '치환어가 같은 꼴인가'만 보면 되므로 어미로 충분하다."""
    x = re.sub(r"[^A-Za-z-]", "", str(w or "")).lower()
    if not x:
        return "?"
    if x.endswith("ing"):
        return "-ing"
    if x.endswith("ied") or (x.endswith("ed") and len(x) > 3):
        return "-ed"
    if x.endswith("ies") or x.endswith("es") or (
            x.endswith("s") and not x.endswith(("ss", "us", "is"))):
        return "-s"
    return "원형"


def shape_mismatch(original: str, shown: str) -> str:
    """치환어가 원문 단어와 형태가 어긋나면 사유 문자열, 맞으면 빈 문자열.

    ★ 형태가 같아도 '-s' 는 복수명사일 수도 3인칭 단수 동사일 수도 있다.
      그건 구분하지 않는다 — 어차피 지문 자리가 같으므로 꼴만 맞으면 된다."""
    a, b = _word_shape(original), _word_shape(shown)
    if a == b:
        return ""
    _nm = {"-s": "3인칭 단수/복수형", "-ed": "과거·과거분사형",
           "-ing": "-ing형", "원형": "원형", "?": "판정 불가"}
    return f"지문은 '{original}'({_nm[a]})인데 shown 이 '{shown}'({_nm[b]})"


def _cap_mismatch(original: str, shown: str) -> str:
    """첫 글자 대소문자가 어긋나면 사유. 문장 첫 단어를 소문자로 내면 문장이 깨진다."""
    o, sh = str(original or "").strip(), str(shown or "").strip()
    if not o or not sh:
        return ""
    if o[:1].isupper() != sh[:1].isupper():
        return (f"지문은 '{o}'({'대문자' if o[:1].isupper() else '소문자'} 시작)인데 "
                f"shown 이 '{sh}'({'대문자' if sh[:1].isupper() else '소문자'} 시작)")
    return ""


def normalize_llm_vocab(raw_items, paragraphs, blank_spans=None,
                        pid: str = "?", report=None) -> Optional[list]:
    """LLM이 준 vocab_items를 검증·보정한다. 못 쓰면 None.

    ★ _s97에서 세 가지를 바꿨다.
      1) 좌표 찾기를 느슨하게 — LLM 이 original 에 'depends.' 처럼 구두점을 붙이거나
         대소문자를 달리 적어도 지문의 그 자리를 찾는다. 찾은 뒤에는 original 을
         **지문의 실제 형태로 덮어쓴다**. 좌표는 출력물과 무관하므로 느슨해도 안전하다.
      2) 형태 검사를 추가 — 지문이 'depends' 인데 shown 이 'rely' 면 본문 수일치가
         깨진다. 코드가 어미를 고치지는 않는다. 어긋난 사실을 report 에 담아
         재시도 프롬프트로 돌려준다.
      3) 하나가 틀렸다고 다섯을 버리지 않는다 — 못 살린 자리만 표시하고, 호출부가
         코드 픽으로 채우거나 사유를 보고 재시도한다.

    report: list 를 넘기면 실패 사유가 문자열로 쌓인다(로그·재시도용).
    """
    _rep = report if isinstance(report, list) else []

    def _fail(msg):
        _rep.append(msg)
        print(f"[VAR][A][{pid}] Q3어휘 normalize — {msg}")
        return None

    if not isinstance(raw_items, list) or len(raw_items) != 5:
        return _fail(f"항목이 {len(raw_items) if isinstance(raw_items, list) else '리스트 아님'}개 (5개여야 함)")
    out = []
    for _no, it in enumerate(raw_items, 1):
        if not isinstance(it, dict):
            return _fail(f"{_no}번 항목이 dict 가 아님")
        p, i = it.get("para"), it.get("idx")
        orig = str(it.get("original", "")).strip()
        if not orig:
            return _fail(f"{_no}번 original 이 비어 있음")
        if not isinstance(p, int) or p < 0 or p >= len(paragraphs):
            return _fail(f"{_no}번 '{orig}' 의 para={p} 가 범위 밖 (0~{len(paragraphs)-1})")
        toks = paragraphs[p][1].split()

        # ── 좌표 찾기: 정확 → 느슨(구두점·대소문자 무시) 순 ──
        def _loose(w):
            return re.sub(r"[^A-Za-z0-9'-]", "", str(w or "")).lower()

        found = None
        if isinstance(i, int) and 0 <= i < len(toks) and toks[i] == orig:
            found = i
        if found is None:                       # ±6칸 정확 일치
            base = i if isinstance(i, int) else 0
            for d in range(0, 7):
                for j in (base - d, base + d):
                    if 0 <= j < len(toks) and toks[j] == orig:
                        found = j; break
                if found is not None:
                    break
        if found is None:                       # 단락 전체 정확 일치, 유일할 때만
            hits = [j for j, w in enumerate(toks) if w == orig]
            if len(hits) == 1:
                found = hits[0]
        if found is None:                       # ★ 느슨 매칭 (_s97)
            lo = _loose(orig)
            hits = [j for j, w in enumerate(toks) if _loose(w) == lo]
            if len(hits) == 1:
                found = hits[0]
            elif len(hits) > 1 and isinstance(i, int):
                found = min(hits, key=lambda j: abs(j - i))
        if found is None:
            _near = " ".join(toks[max(0, (i or 0) - 2):(i or 0) + 3]) if toks else ""
            return _fail(f"{_no}번 '{orig}' 를 para={p} 에서 못 찾음 (idx={i} 근처: '{_near}')")
        i = found
        orig = toks[i]                          # ★ 지문의 실제 형태로 덮어쓴다

        # Q5 빈칸 자리를 덮으면 문항이 깨진다
        if blank_spans:
            b_lo, b_hi = blank_spans.get(p, (-1, -1))
            if b_lo <= i <= b_hi:
                return _fail(f"{_no}번 '{orig}' 가 Q5 빈칸 자리와 겹침 (para={p} idx={i})")
        if "<BLANK" in orig:
            return _fail(f"{_no}번이 빈칸 마커 자체를 잡음")

        _shown = str(it.get("shown", "")).strip() or orig

        # ── ★ 형태 검사 (_s97) — 고치지 않고 사유만 돌려준다 ──
        _sm = shape_mismatch(orig, _shown)
        if _sm:
            return _fail(f"{_no}번 형태 불일치 — {_sm}. 지문 형태 그대로 쓸 것")
        _cm = _cap_mismatch(orig, _shown)
        if _cm:
            return _fail(f"{_no}번 대소문자 불일치 — {_cm}")
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
