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
    # ★ 방향 있는 동사 (_s109) — 어미 목록으로는 안 잡힌다.
    #   실측: 'dissuade' 가 -ade 로 끝나 _GRADABLE_SUFFIX 를 못 통과했다.
    "dissuade", "persuade", "convince", "discourage", "encourage",
    "conceal", "reveal", "expose", "hide", "ignore", "notice", "overlook",
    "disregard", "value", "dismiss", "embrace", "resist", "sustain",
    "maintain", "weaken", "strengthen", "expand", "shrink", "raise", "lower",
    "improve", "worsen", "extend", "limit", "restrict", "release", "confine",
    "capture", "release", "attract", "repel", "unite", "divide", "separate",
    "combine", "simplify", "complicate", "clarify", "obscure", "confirm",
    "deny", "affirm", "doubt", "trust", "suspect", "support", "oppose",
    "advance", "delay", "accelerate", "slow", "begin", "cease", "continue",
    "stop", "allow", "forbid", "permit", "block", "enable", "disable",
    "succeed", "fail", "win", "lose", "rise", "fall", "grow", "decline",
    "seek", "avoid", "pursue", "abandon", "adopt", "reject", "prefer",
    "identify", "confuse", "recognize", "misread", "assert", "concede",
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
    # ★ 사람·집단·역할 (_s109) — 방향이 없다. 무엇으로 바꿔도 논지가 안 뒤집힌다.
    #   실측 실패: 'audience' 가 밑줄 자리로 나갔다. -ence 어미라 _GRADABLE_SUFFIX 를
    #   통과했는데, -ence 는 보통 추상명사(difference, evidence)지 사람 집단이 아니다.
    "audience", "audiences", "reader", "readers", "listener", "listeners",
    "viewer", "viewers", "speaker", "speakers", "writer", "writers",
    "author", "authors", "student", "students", "teacher", "teachers",
    "scientist", "scientists", "researcher", "researchers", "expert", "experts",
    "user", "users", "customer", "customers", "client", "clients",
    "member", "members", "player", "players", "artist", "artists",
    "child", "children", "adult", "adults", "man", "men", "woman", "women",
    "nation", "nations", "country", "countries", "government", "governments",
    "company", "companies", "industry", "industries", "market", "markets",
    # ★ 구체 사물·장소·단위
    "line", "lines", "story", "stories", "word", "words", "sentence", "sentences",
    "page", "pages", "book", "books", "paper", "papers", "article", "articles",
    "region", "regions", "border", "borders", "territory", "territories",
    "land", "lands", "water", "waters", "city", "cities", "town", "towns",
    "brain", "brains", "body", "bodies", "eye", "eyes", "hand", "hands",
    "face", "faces", "head", "heads", "shape", "shapes", "color", "colors",
    "system", "systems", "process", "processes", "method", "methods",
    "result", "results", "reason", "reasons", "idea", "ideas", "topic", "topics",
    "subject", "subjects", "field", "fields", "level", "levels", "step", "steps",
    "stage", "stages", "period", "periods", "moment", "moments", "century",
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

    ★★ 이 함수는 **코드 픽 폴백(pick_vocab_slots)에서만** 쓴다 (_s109).
      LLM 이 고른 것을 되짚어 판정하는 데는 쓰지 않는다 — 어미·목록으로는
      'audience'(통과시키면 안 되는데 -ence 라 통과)와 'dissuade'(막으면 안 되는데
      -ade 라 거부)를 둘 다 틀린다. 목록에 단어를 넣을수록 새는 곳이 늘어난다.
      LLM 의 선택은 `antonym` 칸을 적게 해서 스스로 걸러내게 한다.
      여기는 코드가 직접 고르는 자리라 어림짐작이라도 있어야 한다.

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
        _blocked = blank_spans.get(p_i) or set()   # ★ _s112: 마커 토큰 자리만
        for s_lo, s_hi, _s in _sentences(text):
            sent_no += 1
            # 기출 0/7 — 지문 첫 문장에는 밑줄을 넣지 않는다
            if p_i == 0 and sent_no == 1:
                continue
            for i in range(s_lo, s_hi + 1):
                if i in _blocked:                # Q5 빈칸 마커 자리 회피
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
    #   ★★ _s110 — 이 검사를 뺐다. '흩어라'는 품질 판단이지 기계 확인이 아니다.
    #     짧은 지문은 문장 수가 모자라 어쩔 수 없이 몰리는데 코드는 그걸 구분 못 한다.
    #     프롬프트가 지고, 코드는 개입하지 않는다.
    _ = sents  # (진단용으로만 남긴다)

    # ★ '반대말을 댈 수 있는가'는 LLM 이 antonym 칸에 적어서 증명한다 (_s109).
    #   코드가 어미·목록으로 되짚어 판정하면 양쪽으로 틀린다(audience 통과 / dissuade 거부).
    #   여기서는 그 칸이 채워졌는지만 백스톱으로 본다.
    for it in vocab_items:
        _ant = str(it.get("antonym", "")).strip()
        if not _ant:
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번 '{it.get('original')}' 의 antonym 이 비었다 — "
                f"반대말을 못 적을 단어는 밑줄 자리로 쓰지 마라")

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
        # ★★ _s110 — 철자 유사 판정을 코드에서 뺐다.
        #   difflib 유사도로 재다 보니 'inhabitable/uninhabitable'(_s87), 'modesty'(_s90)
        #   같은 정상 치환을 반복해서 오탐했다. '철자만 비슷한가 뜻이 통하는가'는
        #   의미 판단이라 코드가 못 한다. 프롬프트가 진다.
        #   ★ 형태로 판정 가능한 것(부정 접두사)만 아래에서 계속 본다.
        pass

    # ★ 정답 자리에 부정 접두사 반의어를 쓰면 안 된다 (_s102)
    #   'inhabitable' → 'uninhabitable' 은 철자가 거의 같아 학생이 논지를 안 읽고
    #   'un- 이 붙었네' 로 찍는다. 독해 문항이 철자 찾기가 된다.
    #   ★ CRITICAL 이 아니다 — 이것 하나로 지문을 통째로 버릴 일은 아니다.
    #     엄격 모드에서 재시도만 시키고, 관대 모드에서는 통과시킨다.
    #   ★★ _s115 — 접두사 판정 대신 antonym 칸으로 대조한다.
    #     오답 자리의 shown 이 그 자리 antonym 과 같으면 반의어를 넣은 것이다.
    #     접두사든 어근이 다르든 전부 잡힌다. 순수 문자열 비교라 오탐이 없다.
    _norm = lambda x: re.sub(r"[^a-z]", "", str(x or "").lower())
    for it in vocab_items:
        if it.get("is_answer"):
            continue
        _a, _sh = _norm(it.get("antonym")), _norm(it.get("shown"))
        if _a and _a == _sh:
            errors.append(
                f"[{pid}] [CRITICAL] Q3 어휘 {it.get('n')}번(오답) shown 이 그 자리 "
                f"antonym 과 같다 ('{it.get('shown')}') — 오답 자리는 동의어여야 한다. "
                f"반의어는 정답 자리 하나뿐이다")

    # ★ 선지는 한 단어여야 한다 (_s102)
    #   실측: 'largest' → 'most extensive'(두 단어)가 나갔다. 수능 30번 선지는
    #   예외 없이 한 단어다. 두 단어면 밑줄 길이가 달라져 그 자리가 표가 난다.
    for it in vocab_items:
        sh = str(it.get("shown", "")).strip()
        if len(sh.split()) > 1:
            kind = "정답" if it.get("is_answer") else "오답"
            errors.append(
                f"[{pid}] [CRITICAL] Q3 어휘 {it.get('n')}번({kind}) 제시어가 "
                f"{len(sh.split())}단어 '{sh}' — 선지는 반드시 한 단어여야 한다")
        o = str(it.get("original", "")).strip()
        if len(o.split()) > 1:
            errors.append(
                f"[{pid}] [CRITICAL] Q3 어휘 {it.get('n')}번 원문어가 "
                f"{len(o.split())}단어 '{o}' — 밑줄은 한 단어에만 친다")

    # ★ 형태(굴절) 일치 — 지문이 'depends' 인데 shown 이 'rely' 면 본문 수일치가 깨진다.
    #   normalize 단계에서 이미 걸러지지만, 폴백 경로로 들어온 것도 있으므로 백스톱을 둔다.
    for it in vocab_items:
        # ★ 백스톱은 -s 만 본다(_s100). -ing/-ed 는 형태소로 판정이 안 돼
        #   'vast'→'overwhelming' 같은 정상 치환을 CRITICAL 로 죽인다.
        # ★★ _s111 — 어미 형태 백스톱을 뺐다. 'exciting'→'compelling' 같은
        #   정상 치환을 오탐한다. LLM 이 sentence 칸에 문장을 써서 스스로 검증한다.
        pass

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

    # ★★ shown 끼리도 본다 (_s141). 학생이 보는 건 shown 이다.
    #   실측: 16강 03번에 ② 'relinquishing' 과 ⑤ 'relinquished' 가 같이 나갔다.
    #   original 은 서로 다른 단어('giving' / 'claimed')라 위 검사를 통과했다.
    _shs = [_stem(it.get("shown", "")) for it in vocab_items]
    _sdup = [w for w in set(_shs) if _shs.count(w) > 1 and w]
    if _sdup:
        _sp = [str(it.get("shown", "")) for it in vocab_items
               if _stem(it.get("shown", "")) in _sdup]
        errors.append(f"[{pid}] Q3 어휘 선지에 같은 단어 반복 {_sp} — "
                      f"학생이 보는 다섯 선지는 서로 다른 단어여야 한다 "
                      f"(굴절형도 같은 단어로 본다)")

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

    ★★ _s112 — 금지 구역을 **마커 토큰 그 자리로만** 좁혔다.
      옛 코드는 `(min(hits)-1, max(hits)+1)` 로 잡았는데 문제가 둘이었다.
        (1) 두 마커가 같은 단락에 있으면 **그 사이 전체**가 막힌다.
            `the brain <BLANK_A> ... the vast majority ... <BLANK_B> ...` 에서
            중간의 'majority' 까지 금지 구역이 됐다(실측: A 01번이 이걸로 죽었다).
        (2) 마커 양옆 1토큰도 막았는데 그럴 이유가 없다. 마커는 토큰 하나고
            그 옆은 학생에게 그대로 보이는 지문이다.
      Q5 빈칸 안 단어는 **애초에 지문에서 사라져** LLM 에게 안 보인다
      (프롬프트가 [[[여기는 Q5 빈칸]]] 으로 가린다). 그러니 겹칠 일이 거의 없고,
      막아야 할 것은 마커 자체를 밑줄로 잡는 경우뿐이다.

    반환: {para_index: set(마커 토큰 위치)}
    """
    spans = {}
    for p_i, (_lab, text) in enumerate(paragraphs):
        toks = text.split()
        hits = [i for i, w in enumerate(toks)
                if "<BLANK_A>" in w or "<BLANK_B>" in w]
        if hits:
            spans[p_i] = set(hits)
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


_NEG_PREFIX = ("un", "in", "im", "il", "ir", "dis", "non", "anti", "de", "mis")


def _is_prefix_antonym(a: str, b: str) -> str:
    """부정 접두사만 붙여 만든 반의어인가. 맞으면 접두사, 아니면 빈 문자열.

    ★ 정답 자리에 쓰면 안 된다 (_s102). 'inhabitable' → 'uninhabitable' 처럼
      철자가 거의 같으면 학생이 논지를 안 읽고 'un- 이 붙었네' 로 찍는다.
      독해를 묻는 문항이 철자 찾기가 된다.
    ★ 오답 자리(동의어 치환)에서는 상관없다 — 거기는 방향이 안 뒤집히므로."""
    x = re.sub(r"[^a-z]", "", str(a or "").lower())
    y = re.sub(r"[^a-z]", "", str(b or "").lower())
    if not x or not y or x == y:
        return ""
    lo, hi = (x, y) if len(x) <= len(y) else (y, x)
    for p in _NEG_PREFIX:
        if hi == p + lo:
            return p
    return ""


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
# ★ -ing / -ed 로 끝나지만 굴절형이 아니라 '어근이 그런' 형용사·명사들 (_s100).
#   어미만 보면 'overwhelming' 을 -ing형으로 오판해 'vast' 의 동의어로 못 쓰게 된다.
#   실측: A 01번이 'vast' → 'overwhelming' 때문에 2회 실패하고 지문이 통째로 누락됐다.
_INHERENT_ING = {
    "overwhelming", "interesting", "outstanding", "demanding", "challenging",
    "compelling", "convincing", "striking", "surprising", "promising",
    "misleading", "lasting", "leading", "willing", "cunning", "charming",
    "everything", "nothing", "something", "anything", "morning", "evening",
    "meaning", "being", "thing", "king", "ring", "wing", "spring", "string",
    "long", "along", "among", "young", "strong", "wrong",
}
_INHERENT_ED = {
    "sophisticated", "complicated", "dedicated", "detailed", "limited",
    "advanced", "concerned", "determined", "experienced", "qualified",
    "skilled", "aged", "sacred", "wicked", "naked", "indeed", "need",
    "speed", "deed", "breed", "seed", "creed", "greed", "proceed", "exceed",
    "succeed", "red", "bed", "fed", "led", "wed", "shed", "hundred",
}
_INHERENT_S = {
    "always", "perhaps", "thus", "less", "unless", "across", "various",
    "previous", "obvious", "serious", "curious", "conscious", "precious",
    "analysis", "basis", "crisis", "focus", "status", "bonus", "campus",
    "consensus", "surplus", "process", "access", "success", "excess",
    "progress", "express", "address", "witness", "illness", "business",
}


def _word_shape(w: str) -> str:
    """단어의 굴절 형태를 대략 판정한다.

    ★ 어미만 보면 안 된다(_s100). 'overwhelming' 'sophisticated' 'various' 는
      -ing/-ed/-s 로 끝나지만 굴절형이 아니라 어근이 그런 형용사다.
      목록에 있으면 원형으로 본다. 정확한 형태소 분석은 아니지만,
      '치환어가 같은 꼴인가'만 보면 되므로 이 정도면 충분하다."""
    x = re.sub(r"[^A-Za-z-]", "", str(w or "")).lower()
    if not x:
        return "?"
    if x in _INHERENT_ING or x in _INHERENT_ED or x in _INHERENT_S:
        return "원형"
    if x.endswith("ing") and len(x) > 5:
        return "-ing"
    if x.endswith("ied") or (x.endswith("ed") and len(x) > 4):
        return "-ed"
    if x.endswith("ies") or (x.endswith("es") and len(x) > 4) or (
            x.endswith("s") and len(x) > 3 and not x.endswith(("ss", "us", "is"))):
        return "-s"
    return "원형"


def shape_mismatch(original: str, shown: str, strict: bool = True) -> str:
    """치환어가 원문 단어와 형태가 어긋나면 사유 문자열, 맞으면 빈 문자열.

    ★ -s 와 나머지를 다르게 다룬다 (_s100).
      · **-s 불일치는 항상 막는다.** 여기서 실제로 문장이 깨진다 —
        지문 'the brain depends on' 에 'rely' 를 넣으면 'the brain rely on' 이 된다.
      · **-ing / -ed 불일치는 strict 일 때만 막는다.** 이쪽은 형태소로 판정이 안 된다.
        'overwhelming' 'boring' 'sophisticated' 는 어미가 -ing/-ed 지만 굴절형이 아니라
        어근이 그런 형용사다. 목록으로 막으려 하면 끝없이 샌다
        (실측: 'vast' → 'overwhelming' 이 거부돼 A 01번이 통째로 누락됐다).
        → 첫 시도에는 사유를 돌려주되, 재시도에서는 통과시킨다.
          진짜 깨지는 경우('is adapted to' 에 'adjust')는 드물고,
          그것 때문에 지문을 통째로 버리는 편이 훨씬 손해다."""
    a, b = _word_shape(original), _word_shape(shown)
    if a == b:
        return ""
    if not strict and "-s" not in (a, b):
        return ""                      # 재시도에서는 -ing/-ed 차이를 넘어간다
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
                        pid: str = "?", report=None, strict: bool = True) -> Optional[list]:
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
            if i in (blank_spans.get(p) or set()):
                return _fail(f"{_no}번 '{orig}' 가 Q5 빈칸 마커 자리 (para={p} idx={i}) — "
                             f"그 자리 단어는 이미 사라졌다. 다른 단어를 고를 것")
        if "<BLANK" in orig:
            return _fail(f"{_no}번이 빈칸 마커 자체를 잡음")

        # ★ 구두점 차단은 넣었다가 뺐다 (_s129).
        #   구두점은 **본문에만** 붙고 선지에는 안 붙는다(shown_clean 을 쓴다).
        #     본문   ... a central ④ role, and workers ...
        #     선지   ④ role
        #   원문 그대로라 힌트가 되지 않는다. 막을 이유가 없었다.

        _shown = str(it.get("shown", "")).strip() or orig

        # ★ 선지는 한 단어 (_s102) — 'most extensive' 같은 두 단어가 나갔다
        if len(_shown.split()) > 1:
            return _fail(f"{_no}번 제시어가 {len(_shown.split())}단어 '{_shown}' — "
                         f"선지는 한 단어여야 한다")
        # ★ 부사 차단은 넣었다가 뺐다 (_s131).
        #   품사가 아니라 **방향이 기준**이다. 방향 있는 부사는 정답으로 쓸 수 있다 —
        #     rarely ↔ frequently   willingly ↔ reluctantly   naturally ↔ artificially
        #   방향 없는 부사(mentally / understandably / clearly)는 antonym 대입 검증이
        #   걸러낸다 — 반대말을 넣어도 원문과 정반대 주장이 안 되기 때문이다.
        #   품사로 막으니 오답 자리까지 걸려 A 가 죽었다(_s129 실측).

        # ★★ 기능어는 밑줄 대상이 아니다 (_s135).
        #   관사·전치사·대명사·의문사·조동사는 방향이 없어 반대말을 댈 수 없다.
        #   실측: 'why?' 가 선지로 나갔다 — 의문사라 반의어가 성립하지 않는다.
        #   _VOCAB_STOP 은 코드 픽(vocab_candidates)과 answer_pos_ok 에만 걸려 있었고,
        #   answer_pos_ok 는 _s109 에서 안 쓰게 돼 **LLM 픽은 아무도 안 막았다.**
        #   ★ 이건 닫힌 목록이라 형태 판정처럼 끝없이 새지 않는다.
        _bare_o = re.sub(r"[^A-Za-z-]", "", orig).lower()
        if _bare_o in _VOCAB_STOP:
            _k = "정답" if it.get("is_answer") else "오답"
            return _fail(f"{_no}번({_k}) '{orig}' 는 기능어(관사·전치사·대명사·의문사·"
                         f"조동사) — 방향이 없어 반대말을 댈 수 없다. "
                         f"내용어(형용사·동사·명사)에서 고를 것")

        # ★★ 방향 없는 구체명사도 막는다 (_s140).
        #   _CONCRETE 는 코드 픽(_looks_gradable)에서만 쓰고 LLM 픽은 안 봤다 —
        #   _s109 에서 answer_pos_ok 를 안 쓰게 하면서 같이 빠졌다.
        #   실측: 능률(오) 2과 4번에 'years' 'people' 이 밑줄로 나갔다.
        #   사람·집단·시간·구체사물은 무엇으로 바꿔도 논지가 안 뒤집힌다.
        #   ★ 닫힌 목록이라 형태 판정처럼 끝없이 새지 않는다.
        if _bare_o in _CONCRETE:
            _k = "정답" if it.get("is_answer") else "오답"
            return _fail(f"{_no}번({_k}) '{orig}' 는 방향 없는 구체명사 — "
                         f"사람·집단·시간·사물은 무엇으로 바꿔도 논지가 안 뒤집힌다. "
                         f"반대말을 댈 수 있는 말로 고를 것")

        # ★★ 접속부사·담화표지는 밑줄 대상이 아니다 (_s103)
        #   'Similarly,' 'Conversely,' 는 논리 흐름 표지지 문맥 판단 대상이 아니다.
        #   옛 코드는 validate_vocab 에서만 잡아 CRITICAL 을 냈다 — 앞문은 열고
        #   뒷문을 잠근 꼴이라 사유가 재시도 프롬프트로 안 넘어가 같은 단어를
        #   두 번 골랐고 A 02번이 통째로 누락됐다. 여기서 막아 사유를 돌려준다.
        if is_discourse_marker(orig):
            _k = "정답" if it.get("is_answer") else "오답"
            return _fail(f"{_no}번({_k}) '{orig}' 는 접속부사·담화표지 — "
                         f"논리 흐름 표지라 문맥 판단 대상이 아니다. "
                         f"밑줄은 방향을 가진 형용사·동사·명사에만 친다")

        # ★★ '반대말을 댈 수 있는가'는 의미 판단이다 — 코드가 못 한다 (_s109).
        #   옛 코드는 answer_pos_ok 로 어미·목록을 보고 되짚어 판정했는데 양쪽으로 틀렸다:
        #     'audience'  → -ence 어미라 통과 (사람 집단이라 방향이 없는데)
        #     'dissuade'  → -ade 라 목록에 없어 거부 (방향이 명백한 동사인데)
        #   목록에 단어를 넣을수록 새는 곳이 늘어난다 — audience 넣으면 spectator 가 남는다.
        #   → **LLM 이 다섯 자리 전부에 반대말을 적게 하고, 코드는 적혔는지만 본다.**
        #     못 적는 자리는 LLM 이 스스로 버린다. 의미 판단은 LLM, 확인은 코드.
        _ant = str(it.get("antonym", "")).strip()
        if not _ant:
            return _fail(f"{_no}번 '{orig}' 의 antonym 이 비었다 — "
                         f"반대말을 못 적을 단어는 밑줄 자리로 쓰지 마라 "
                         f"(audience/story/region 처럼 방향 없는 말). 다른 문장을 고를 것")
        if re.sub(r"[^a-z]", "", _ant.lower()) == re.sub(r"[^a-z]", "", orig.lower()):
            return _fail(f"{_no}번 antonym 이 원문어와 같다 ('{orig}') — 반대말을 적어라")
        if len(_ant.split()) > 1:
            return _fail(f"{_no}번 antonym 이 {len(_ant.split())}단어 '{_ant}' — 한 단어로 적어라")

        # ★ 부정 접두사 붙이기·떼기 금지 (_s102)
        #   ★★ _s115 — 부정 접두사 판정을 뺐다.
        #     'un-/in- 이 붙었나'는 형태로 보이지만 실제 판단은 '철자로 답이 새는가'라
        #     의미 판단이다. 게다가 진짜 문제의 일부만 잡는다 —
        #       오답에 inhabitable→habitable  (접두사라 잡힘)
        #       오답에 significant→trivial    (반의어인데 못 잡음)
        #     ★ 대신 antonym 칸으로 기계 대조한다. 오답 자리의 shown 이 그 자리의
        #       antonym 과 같으면 반의어를 넣은 것이다 — 접두사든 아니든 다 잡힌다.
        if not it.get("is_answer"):
            _n2 = lambda x: re.sub(r"[^a-z]", "", str(x or "").lower())
            if _n2(_ant) and _n2(_ant) == _n2(_shown):
                return _fail(f"{_no}번(오답) shown 이 그 자리 antonym 과 같다 "
                             f"('{_shown}') — 오답 자리는 **동의어**여야 한다. "
                             f"반의어는 정답 자리 하나뿐이다")

        # ── ★ 형태 검사 (_s97) — 고치지 않고 사유만 돌려준다 ──
        # ★★ 어미로 형태를 판정하지 않는다 (_s111).
        #   'exciting'→'compelling', 'interesting.'→'boring.', 'desirable'→'coveted'
        #   전부 정상 치환인데 -ing/-ed 어미 때문에 오탐했다. 형태소로는 굴절형인지
        #   어근이 그런 형용사인지 구분이 안 된다.
        #   → LLM 이 `sentence` 칸에 치환 문장을 써서 낸다. 비문이면 써 보는 순간 보인다.
        #   코드는 그 문장이 **원문 문장과 한 단어만 다른가**만 대조한다(기계 확인).
        #   ★★ _s114 — sentence 칸을 없앴다.
        #     어휘 문제는 **단어 하나만 바꾸는 것**이다. 문장을 쓰게 하니 '그 문장이
        #     어디까지인가'라는 새 문제가 생겼고(_s111 에서 A 3/3 전멸), 그걸 완화하니
        #     껍데기 검사만 남았다(_s113). 애초에 필요 없는 칸이었다.
        #     형태가 맞는지는 LLM 이 shown 을 고를 때 판단한다 —
        #     코드는 대소문자만 본다(아래).

        # ★ 대소문자만 코드가 본다 — 문장 첫 단어를 소문자로 내면 문장이 깨진다.
        _cm = _cap_mismatch(orig, _shown)
        if _cm:
            return _fail(f"{_no}번 대소문자 불일치 — {_cm}")
        out.append({
            "n": it.get("n"), "para": p, "idx": i, "original": orig,
            "shown": _shown,
            # ★ antonym 을 반드시 실어 보낸다 (_s119).
            #   빠뜨렸더니 validate_vocab 이 "antonym 이 비었다"를 매번 냈다 —
            #   normalize 는 통과했는데 뒤에서 죽는 구조였다(실측 40건, A 3/3 관대 모드).
            "antonym": _ant,
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
