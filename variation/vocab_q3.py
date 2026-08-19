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
      막아야 할 것은 방향 없는 구체명사(tasks, time, readers)뿐이다.

    ★ _CONCRETE 검사는 여기서 뺐다 (_s142). 관문 blocked_reason() 이 먼저 보므로
      여기서 또 보면 같은 규칙이 두 곳에 생긴다 — 그게 _s135·_s140 의 원인이었다.
      이 함수는 **목록 차단이 아니라 '방향이 있어 보이는가' 어림짐작만** 한다."""
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
                # ★ 목록 차단은 관문 한 곳에서만 (_s142) — 기능어·담화표지·구체명사
                if blocked_reason(bare):
                    continue
                # ★ 구동사 자리는 쓰지 않는다 (_s146) — 동사만 바꾸면 불변화사가 남는다
                if i + 1 <= s_hi and i + 1 < len(toks) and particle_follows(toks[i + 1]):
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


# ════════════════════════════════════════════════════════════════
# ★ 부정관사 a/an 보정 (_s143)
#   어휘 유형은 본문 단어를 shown 으로 바꿔 보여준다. 앞에 부정관사가 있으면
#   첫소리가 바뀌면서 관사가 안 맞을 수 있다.
#   실측: 'an enclosed space' → 'an ⑤ sealed space' (공통영어2 YBM(박) 01과 FR).
#   학생 눈에 바로 보이는 비문이고, "여기가 바뀐 자리"라는 힌트까지 된다.
#   ★ a/an 은 철자가 아니라 **소리** 규칙이라 코드가 다 맞힐 수 없다
#     (an hour / a university). 확실한 것만 고치고 애매하면 손대지 않는다 —
#     이 코드베이스의 원칙 그대로다(닫힌 목록만 코드가 본다).
# ════════════════════════════════════════════════════════════════

# 자음자로 시작하지만 'an' 을 쓰는 말 (묵음 h)
_AN_DESPITE_CONSONANT = {
    "hour", "hours", "hourly", "honest", "honestly", "honesty",
    "honor", "honors", "honored", "honorable", "honour", "honours",
    "heir", "heirs", "heiress",
}
# 모음자로 시작하지만 'a' 를 쓰는 말 (반모음 /j/·/w/)
_A_DESPITE_VOWEL = {
    "university", "universities", "universal", "universally", "unique", "uniquely",
    "unit", "units", "united", "union", "unions", "uniform", "unified", "unifying",
    "useful", "usefully", "user", "users", "usage", "usual", "usually",
    "utility", "utilities", "european", "europe", "euphemism", "eulogy",
    "one", "once", "ubiquitous", "unanimous", "unilateral",
}


def needs_an(word: str):
    """앞에 'an' 이 와야 하면 True, 'a' 면 False, **확신이 없으면 None**.

    None 일 때는 관사를 건드리지 않는다. 잘못 고치면 원래보다 나쁘다.
    """
    w = re.sub(r"^[^A-Za-z]+", "", str(word or ""))
    if not w:
        return None
    low = w.lower()
    if low in _AN_DESPITE_CONSONANT:
        return True
    if low in _A_DESPITE_VOWEL:
        return False
    # 약어·한 글자는 알파벳 이름으로 읽어서 규칙이 다르다 (an FBI, a UN) — 손대지 않는다
    if len(w) == 1 or w.isupper():
        return None
    return low[0] in "aeiou"


def article_mismatch(prev_token: str, shown: str) -> bool:
    """앞 토큰이 부정관사인데 shown 과 안 맞으면 True."""
    art = re.sub(r"[^A-Za-z]", "", str(prev_token or "")).lower()
    if art not in ("a", "an"):
        return False
    need = needs_an(shown)
    if need is None:
        return False
    return (art == "an") != need


def _fix_article_token(prev_token: str, shown: str) -> str:
    """앞 토큰(부정관사)을 shown 에 맞게 고쳐 돌려준다. 대문자·구두점 유지."""
    if not article_mismatch(prev_token, shown):
        return prev_token
    want = "an" if needs_an(shown) else "a"
    def _swap(m):
        w = m.group(0)
        return want.capitalize() if w[0].isupper() else want
    return re.sub(r"[A-Za-z]+", _swap, prev_token, count=1)


# 단어 끝에 붙는 구두점 — 밑줄 밖으로 뺀다 (_s144)
_TRAIL_PUNCT = re.compile(r'[.,;:!?)\]"\u2019\u201d]+$')


# ★ 조동사로 바꾸면 뒤의 to 가 남는다 (_s145)
#   실측: 원문 'Participants need to be…' 에서 need → must 로 바꿔
#   'Participants must to be…' 가 됐다(25년 고1 9월 28번 ⑤).
#   조동사 뒤에는 to 가 오지 않는다. ⑤는 오답 자리인데 학생 눈에 명백한
#   비문이라 "여기가 답"으로 찍게 만든다 — 정답 시비가 나는 자리다.
#   ★ 원문 대조로는 안 잡힌다. shown 을 original 로 되돌리면 원문과 같아지기
#     때문이다. 화면에 보이는 문장을 따로 봐야 한다 — 관사(_s143)와 같은 부류다.
_BARE_MODALS = {"must", "can", "could", "will", "would", "shall", "should",
                "may", "might"}


# ★ 구동사 불변화사 — 동사만 바꾸면 이게 남아 비문이 된다 (_s146)
#   실측: 원문 'These bacteria can break down plastic.' 에서 break → decompose 로
#   바꿔 'can decompose down plastic' 이 됐다(능률(민) 04과 02번 ④).
#   같은 사고가 _s141 에도 있다 — 'giving up' → 'relinquishing up'.
#   그때는 프롬프트에만 적고 검사 코드를 안 넣어서 또 샜다.
#   ★ up/down/off/out/away/back 만 본다. in/on/over/about 은 평범한 전치사로
#     쓰이는 일이 훨씬 많아 넣으면 멀쩡한 자리까지 걷어낸다.
#   ★ 실측(실지문 12개 220후보): 2개(0.9%)만 줄고 5개 미만이 되는 지문은 없다.
_PARTICLES = {"up", "down", "off", "out", "away", "back"}


def particle_follows(next_token: str) -> bool:
    """바로 뒤 단어가 구동사 불변화사인가."""
    return re.sub(r"[^A-Za-z]", "", str(next_token or "")).lower() in _PARTICLES


def modal_before_to(shown: str, next_token: str) -> bool:
    """shown 이 조동사인데 바로 뒤가 to 면 비문('must to be')."""
    w = re.sub(r"[^A-Za-z]", "", str(shown or "")).lower()
    nx = re.sub(r"[^A-Za-z]", "", str(next_token or "")).lower()
    return w in _BARE_MODALS and nx == "to"


def _trailing_punct(token: str) -> str:
    m = _TRAIL_PUNCT.search(str(token or ""))
    return m.group(0) if m else ""


def _strip_trailing_punct(w: str) -> str:
    return _TRAIL_PUNCT.sub("", str(w or ""))


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
                # ★ 앞의 부정관사가 안 맞으면 같이 고친다 (_s143)
                #   'an enclosed' → 'an sealed' 같은 비문을 막는다.
                if i > 0 and article_mismatch(toks[i - 1], shown):
                    toks[i - 1] = _fix_article_token(toks[i - 1], shown)
                # ★ 원문 토큰의 끝 구두점을 밑줄 **밖**에 붙인다 (_s144).
                #   (1) 문장 끝 마침표가 사라지던 것을 막는다 —
                #       실측 'use them to communicate.' → 'use them to interact'
                #       (25년 고1 9월 26번). 순서대로 이으면 문장이 안 끊겼다.
                #   (2) shown 에 구두점이 붙어 올 때마다 밑줄이 구두점까지 덮었다
                #       ('responded.' 'baseless.'). 기출은 단어에만 긋는다.
                #   구두점은 원문 것을 쓴다 — 문장 구조는 원문이 정한다.
                tail = _trailing_punct(it["original"])
                core = _strip_trailing_punct(shown)
                toks[i] = (f'<span class="vmark">{marks[it["n"] - 1]}</span>'
                           f'<u class="vword">{core}</u>{tail}')
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

    # 앞의 부정관사와 안 맞는 shown (_s143)
    #   apply_vocab_items 가 렌더 때 고치지만, 여기서도 본다 — 고쳐지지 않은 채로
    #   올라온 옛 캐시를 잡아내야 하고, 검사가 렌더 한 곳에만 있으면
    #   _s135·_s140 과 같은 구조가 다시 생긴다.
    for it in vocab_items:
        p_i, i = it.get("para"), it.get("idx")
        if not isinstance(p_i, int) or not isinstance(i, int) or p_i >= len(paragraphs):
            continue
        toks = paragraphs[p_i][1].split()
        if i <= 0 or i >= len(toks):
            continue
        shown = it.get("shown") or it.get("original") or ""
        if i + 1 < len(toks) and particle_follows(toks[i + 1]):
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번 '{shown} {toks[i+1]}' — 원문 "
                f"'{it.get('original')} {toks[i+1]}' 는 구동사다. 동사만 바꾸면 "
                f"불변화사가 남아 비문이 된다")
        if i + 1 < len(toks) and modal_before_to(shown, toks[i + 1]):
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번 '{shown} to' — 조동사 뒤에는 to 가 "
                f"오지 않는다. 원문 '{it.get('original')} to' 를 조동사로 바꾸면 비문이 "
                f"된다 (have/need 처럼 to 를 받는 말로 고를 것)")
        if article_mismatch(toks[i - 1], shown):
            want = "an" if needs_an(shown) else "a"
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번 '{toks[i-1]} {shown}' — "
                f"부정관사가 안 맞는다. '{want} {shown}' 이어야 한다")

    # 밑줄로 쓸 수 없는 단어 — 관문 한 곳에서 본다 (_s142)
    #   ★ 여기는 백스톱이다. 관대 모드로 통과했거나 옛 캐시에서 올라온 항목은
    #     normalize 를 안 거치고 여기로 바로 온다. 예전엔 이 자리가 담화표지
    #     하나만 봐서, 'why?'·'years' 같은 건 여기서도 안 걸렸다.
    for it in vocab_items:
        o = str(it.get("original", ""))
        if not o:
            continue
        why = blocked_reason(o)
        if why:
            kind = "정답" if it.get("is_answer") else "오답"
            errors.append(
                f"[{pid}] Q3 어휘 {it.get('n')}번({kind}) '{o}'는 {why}")

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


def _sentence_bounds(toks) -> list:
    """토큰 리스트를 문장 단위 (시작, 끝) 구간으로 나눈다 (_s148).

    약어의 마침표는 문장 끝으로 보지 않는다 — 'p.m.' 'Ph.D.' 'U.S.' 등.
    """
    out, lo = [], 0
    for i, w in enumerate(toks):
        bare = re.sub(r'[")\]\u2019\u201d]+$', "", w)
        if not re.search(r'[.!?]$', bare):
            continue
        if re.search(r'\b(?:[ap]\.m|u\.s|u\.k|u\.n|ph\.d|e\.g|i\.e|'
                     r'dr|mr|mrs|ms|prof|jr|sr|st|vs|etc|no|vol|fig)\.$', bare, re.I):
            continue
        if re.fullmatch(r'[A-Za-z]\.', bare):      # 1글자 이니셜
            continue
        out.append((lo, i)); lo = i + 1
    if lo < len(toks):
        out.append((lo, len(toks) - 1))
    return out


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
    """문두 접속부사·담화표지인가.

    ★ 밑줄 자격 판정에는 이 함수를 직접 쓰지 말 것 (_s142).
      blocked_reason() 이 담화표지를 포함해 전부 본다. 여기만 따로 부르면
      검사대가 다시 둘로 갈라진다 — 그게 _s135·_s140 이 새어나간 구조다.
      (지금은 blocked_reason 안에서만 같은 목록을 본다.)"""
    return strip_edge_punct(w).lower() in _DISCOURSE_MARKER


# ════════════════════════════════════════════════════════════════
# ★★ 밑줄 단어 관문 (_s142) — 목록으로 막는 검사는 전부 여기 한 곳에서만 한다
#
# 밑줄 칠 단어가 정해지는 길은 셋이다:
#   ① 코드가 직접 고르는 길   vocab_candidates / pick_vocab_slots (LLM 실패 시 폴백)
#   ② LLM 이 골라온 길        normalize_llm_vocab   ← 평소에 쓰는 길
#   ③ 마지막 확인             validate_vocab        ← 캐시·관대모드 백스톱
#
# 예전엔 같은 검사가 이 셋에 흩어져 있었다. 그래서 한쪽을 손대면 다른 쪽이
# 조용히 비었고, 목록은 파일에 그대로 남아 있어 겉보기엔 막혀 있는 것처럼 보였다:
#   · _s109 에서 answer_pos_ok 를 안 쓰게 하자 ② 의 목록 검사가 통째로 사라졌다.
#   · _s135 — 'why?' 가 선지로 나갔다 (_VOCAB_STOP 이 ① 에만 걸려 있었다).
#   · _s140 — 'years' 'people' 이 밑줄로 나갔다 (_CONCRETE 가 ① 에만 걸려 있었다).
#   다섯 버전을 사이에 두고 **같은 원인으로 두 번** 터졌고, 그때마다 한 항목씩
#   손으로 ② 에 옮겨 붙였다. ③ 은 아직도 담화표지 하나만 보고 있었다.
# → 이제 셋 다 blocked_reason() 을 지나간다. 목록에 단어를 넣으면 세 길에 동시에
#   걸린다. "어느 검사대에 붙였더라"를 기억할 필요가 없다.
#
# ★ 여기 두지 않는 것: '반대말을 댈 수 있는가' 같은 의미 판단 (_s109·_s131).
#   코드가 어미·품사로 재면 'audience' 는 통과하고(-ence) 'dissuade' 는 거부한다
#   (-ade) — 양쪽으로 틀린다. 목록에 단어를 넣을수록 새는 곳이 늘어난다.
#   그건 LLM 이 antonym 칸을 채우게 해서 거른다. 여기는 **닫힌 목록만** 본다.
# ════════════════════════════════════════════════════════════════

# 밑줄 자리로 쓰기엔 너무 짧은 단어 (a, an, of … 는 _VOCAB_STOP 이 잡지만
# 목록에 없는 2글자 토큰도 변별력이 없다)
_MIN_WORD_LEN = 3


def blocked_reason(word: str) -> Optional[str]:
    """밑줄(선지) 자리로 쓸 수 없는 단어면 사유 문자열, 쓸 수 있으면 None.

    사유 문자열은 그대로 LLM 재시도 프롬프트에 넘어간다 — 무엇을 어떻게
    고쳐야 하는지가 문장에 들어 있어야 한다(사유가 안 넘어가면 같은 단어를
    다시 골라 재시도가 소진된다. _s103 에서 실제로 그렇게 A 02번이 누락됐다).
    """
    bare = re.sub(r"[^A-Za-z-]", "", str(word or "")).lower()

    if not bare or len(bare) < _MIN_WORD_LEN:
        return ("너무 짧은 말이라 변별력이 없다 — "
                "방향을 가진 형용사·동사·명사에서 고를 것")

    if bare in _VOCAB_STOP:
        return ("기능어(관사·전치사·대명사·의문사·조동사) — "
                "방향이 없어 반대말을 댈 수 없다. "
                "내용어(형용사·동사·명사)에서 고를 것")

    if bare in _DISCOURSE_MARKER:
        return ("접속부사·담화표지 — 논리 흐름 표지라 문맥 판단 대상이 아니다. "
                "밑줄은 방향을 가진 형용사·동사·명사에만 친다")

    if bare in _CONCRETE:
        return ("방향 없는 구체명사 — 사람·집단·시간·사물은 무엇으로 바꿔도 "
                "논지가 안 뒤집힌다. 반대말을 댈 수 있는 말로 고를 것")

    return None


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


# ★★ answer_pos_ok() 는 없앴다 (_s142).
#   _s109 에서 "쓰지 않는다"로 정하고도 함수는 남겨뒀다. 그 안에 _VOCAB_STOP·
#   _DISCOURSE_MARKER 검사가 들어 있어서, 코드를 열어보면 막고 있는 것처럼 보이는데
#   실제로는 아무도 부르지 않았다 — 이 '살아 있는 척하는 죽은 검사대'가
#   _s135·_s140 을 못 알아챈 이유다. 목록 검사는 blocked_reason() 이 전부 한다.
#   (품사·어미로 정답 자격을 재던 부분은 되살리지 않는다 — _s109·_s131 참조.)


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

        # ★★ 목록 차단은 관문 한 곳에서 (_s142).
        #   예전엔 기능어(_s135)·구체명사(_s140)·담화표지(_s103) 검사가 여기에
        #   따로따로 붙어 있었다. 코드 픽 쪽과 짝이 안 맞아 한쪽만 비는 일이
        #   반복됐다 — 'why?'(_s135), 'years' 'people'(_s140) 이 그렇게 새어나갔다.
        #   이제 blocked_reason() 하나만 부른다. 사유는 그대로 재시도로 넘어간다.
        _blocked = blocked_reason(orig)
        if _blocked:
            _k = "정답" if it.get("is_answer") else "오답"
            return _fail(f"{_no}번({_k}) '{orig}' 는 {_blocked}")

        # ★ 조동사로 바꾸면 뒤의 to 가 남는다 (_s145) — 'need to' → 'must to'.
        #   여기서 막아야 사유가 재시도 프롬프트로 넘어간다(_s103 의 교훈).
        try:
            _tk = paragraphs[int(it.get("para"))][1].split()
            _ix = int(it.get("idx"))
            # ★★ idx 는 LLM 이 스스로 적은 값이라 틀릴 수 있다 (_s156).
            #   실측(25년 고1 9월 23번): original='clear'(단락 안 21번째)인데 LLM 이
            #   27을 적었다. 27 뒤는 'away' 라 코드가 'clear away 는 구동사다'로
            #   거부했다 — 지문엔 'clear view' 다. 멀쩡한 자리가 세 번 버려졌다.
            #   → 그 자리 낱말이 original 과 다르면, 단락에서 찾아 바로잡는다.
            #     유일하게 못 찾으면 이 검사는 건너뛴다(틀린 자리로 거부하지 않는다).
            def _bare(w):
                return re.sub(r"[^A-Za-z']", "", str(w or "")).lower()
            if not (0 <= _ix < len(_tk)) or _bare(_tk[_ix]) != _bare(orig):
                _hits = [k for k, w in enumerate(_tk) if _bare(w) == _bare(orig)]
                if len(_hits) == 1:
                    _ix = _hits[0]
                else:
                    raise _IdxUnknown
            if _ix + 1 < len(_tk) and particle_follows(_tk[_ix + 1]):
                return _fail(f"{_no}번 '{orig} {_tk[_ix + 1]}' 는 구동사다 — 동사만 바꾸면 "
                             f"'{it.get('shown')} {_tk[_ix + 1]}' 처럼 불변화사가 남아 비문이 된다. "
                             f"다른 자리를 고를 것")
            if _ix + 1 < len(_tk) and modal_before_to(it.get("shown"), _tk[_ix + 1]):
                return _fail(f"{_no}번 '{it.get('shown')} to' — 조동사 뒤에는 to 가 "
                             f"오지 않는다. 원문이 '{orig} to' 이므로 to 를 받는 말"
                             f"(have/need/ought 등)로 고르거나 다른 자리를 고를 것")
        except _IdxUnknown:
            pass          # 자리를 못 정하면 이 검사만 건너뛴다 (_s156)
        except Exception:
            pass

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

# ════════════════════════════════════════════════════════════════
# ★ Q3 정답 자리와 Q4 진술이 겹치면 정답이 갈린다 (_s147)
#   Q4 진술·해설은 **원문 기준**으로 만들어지는데, 학생이 보는 지문은
#   Q3 어휘로 바뀐 것이다. 동의어로 바뀐 자리는 판정이 안 갈리지만
#   **Q3 정답 자리(반의어로 뒤집힌 곳)** 가 Q4 근거와 같으면 정반대가 된다.
#   실측(능률(민) 04과 01번): 지문에 'Achieving these goals seems easy' 로
#   찍혀 있는데 Q4 진술 '라'가 "…seems easy" 이고 해설은 원문 'difficult' 를
#   근거로 (X) 거짓이라 했다. 학생이 지문대로 읽으면 참이다 — 정답 시비.
#   덤으로 Q3 정답을 Q4 쪽에서 역산할 수도 있다(두 문항이 서로 답을 흘린다).
# ════════════════════════════════════════════════════════════════

def _key_words(sent: str) -> set:
    """비교용 내용어 집합 (기능어·구두점 제거)."""
    ws = re.sub(r"[^A-Za-z\s]", " ", str(sent or "")).lower().split()
    return {w for w in ws if len(w) > 3 and w not in _VOCAB_STOP}


def q4_conflicts_with_answer(vocab_items, paragraphs, statements_evidence) -> list:
    """Q3 정답 자리 문장을 근거로 삼은 Q4 진술이 있으면 그 라벨 목록.

    statements_evidence: Q4 각 진술의 근거 문장 리스트(원문 문장).
    """
    ans = next((it for it in vocab_items or [] if it.get("is_answer")), None)
    if not ans:
        return []
    try:
        toks = paragraphs[int(ans["para"])][1].split()
        i = int(ans["idx"])
    except Exception:
        return []
    # 정답 단어가 든 문장을 원문에서 떼어낸다
    lo = i
    while lo > 0 and not re.search(r"[.!?]$", toks[lo - 1]):
        lo -= 1
    hi = i
    while hi < len(toks) - 1 and not re.search(r"[.!?]$", toks[hi]):
        hi += 1
    sent = " ".join(toks[lo:hi + 1])
    key = _key_words(sent)
    if len(key) < 3:
        return []
    hits = []
    for n, ev in enumerate(statements_evidence or []):
        ek = _key_words(ev)
        if not ek:
            continue
        # 근거 문장이 정답 문장과 사실상 같은가 (내용어 70% 이상 겹침)
        if len(key & ek) / max(1, min(len(key), len(ek))) >= 0.7:
            hits.append("가나다라마"[n] if n < 5 else str(n + 1))
    return hits

# ════════════════════════════════════════════════════════════════
# ★ Q3 정답 자리는 Q5 빈칸이 든 문장을 피한다 (_s148)
#   실측(영어2 능률(오) 01과 03번):
#     'To ③hinder them, local doctors and nurses <BLANK_B>.'
#   밑줄과 마커가 다른 토큰이라 _s112 규칙은 안 어겼다. 그런데 술부가 통째로
#   빈칸이라 hinder 가 틀렸는지 판단할 근거가 그 문장에 없다.
#   ★ 금지 구역을 문장 단위로 넓히면 후보가 33% 줄어든다(실측 12지문).
#     그건 _s112 가 되돌린 실수를 반복하는 것이다. 그래서 **정답 자리에만**
#     건다 — 오답 넷은 동의어라 판단할 게 없으므로 빈칸 문장에 있어도 무해하다.
# ════════════════════════════════════════════════════════════════

def answer_in_blank_sentence(vocab_items, paragraphs) -> bool:
    """Q3 정답 단어가 Q5 빈칸 마커와 같은 문장에 있으면 True."""
    ans = next((it for it in vocab_items or [] if it.get("is_answer")), None)
    if not ans:
        return False
    try:
        toks = paragraphs[int(ans["para"])][1].split()
        i = int(ans["idx"])
    except Exception:
        return False
    for lo, hi in _sentence_bounds(toks):
        if lo <= i <= hi:
            return any("<BLANK_A>" in toks[k] or "<BLANK_B>" in toks[k]
                       for k in range(lo, hi + 1))
    return False

# ════════════════════════════════════════════════════════════════
# ★ Q4 진술이 Q5 빈칸 정답을 그대로 말해주면 안 된다 (_s148)
#   오늘 검수에서 제일 자주 난 유형이다(16지문 중 6건).
#     진술 "Ocean Alliance was founded to protect whales and the oceans"
#     ↔ Q5(A) 정답 'founded Ocean Alliance to protect whales and the earth's oceans'
#   Q4 를 먼저 읽으면 Q5 영작 답이 보인다. 배점 하나가 공짜가 된다.
#   ★ 판정은 **내용어 4연속 일치** 로 한다. 낱말 몇 개 겹치는 것은 같은 지문이니
#     당연하고, 그걸로 막으면 멀쩡한 진술까지 걷어낸다.
# ════════════════════════════════════════════════════════════════

def _content_grams(text: str, k: int = 4) -> set:
    ws = [w for w in re.findall(r"[A-Za-z']+", str(text or "").lower())]
    return {tuple(ws[i:i + k]) for i in range(len(ws) - k + 1)}


def statements_leak_blanks(statements, blank_a: str, blank_b: str) -> list:
    """Q5 정답을 4단어 이상 그대로 담은 Q4 진술의 라벨 목록.

    statements: [(라벨, 문장, 참/거짓), ...] 또는 문자열 리스트
    """
    out = []
    grams = [( "A", _content_grams(blank_a)), ("B", _content_grams(blank_b))]
    for n, st in enumerate(statements or []):
        txt = st[1] if isinstance(st, (list, tuple)) and len(st) > 1 else st
        g = _content_grams(txt)
        if not g:
            continue
        for tag, bg in grams:
            if bg and (g & bg):
                lab = "가나다라마"[n] if n < 5 else str(n + 1)
                out.append((lab, tag))
                break
    return out

# ════════════════════════════════════════════════════════════════
# ★ Q4 근거 인용문이 지문에 실제로 있는가 (_s149)
#   Q4 의 O/X 판정이 틀리는 사고가 있었다. 판정 자체("이 진술이 지문과 맞나")는
#   의미 판단이라 코드가 못 한다. 그런데 그 앞 단계는 잡힌다 —
#   **근거로 든 문장이 지문에 있기는 한가.** 지어낸 근거는 판정도 대개 틀린다.
#   ★ 글자 그대로 비교하지 않는다. LLM 이 인용할 때 'and eventually' →
#     'he eventually' 처럼 살짝 바꾸거나 따옴표 종류를 바꾼다(실측 2건, 둘 다
#     정상 인용이었다). 내용어 4연속 일치로 본다 — 지어낸 근거는 어느 4연속도
#     안 맞고, 살짝 바꾼 인용은 다른 자리에서 맞는다.
# ════════════════════════════════════════════════════════════════

def evidence_not_in_passage(statements, evidences, passage_text) -> list:
    """지문에 없는 근거를 든 진술의 라벨 목록."""
    pool = _content_grams(passage_text)
    if not pool:
        return []
    out = []
    for i, ev in enumerate(evidences or []):
        if not isinstance(ev, str) or len(ev.split()) < 5:
            continue
        if not (_content_grams(ev) & pool):
            lab = ""
            try:
                lab = str(statements[i][0])
            except Exception:
                lab = "가나다라마"[i] if i < 5 else str(i + 1)
            out.append((lab, ev.strip()[:70]))
    return out

# ════════════════════════════════════════════════════════════════
# ★ 내부 마커가 산출물로 새어나가면 안 된다 (_s150)
#   프롬프트는 Q5 빈칸 자리를 [[[여기는 Q5 빈칸 …]]] 로 가리고, 지문에는
#   <BLANK_A>/<BLANK_B> 마커를 심는다. 전부 **코드 내부용**이다.
#   실측(25년 고2 9월 37·39번): LLM 이 그 가림 문자열을 근거 인용문에 그대로
#   베껴 답지에 찍혔다 — 근거 “… the tension force is too large for the nail,
#   or [[[...]]]”. 선생님·학생이 보는 답지에 내부 문자열이 나온다.
#   ★ 닫힌 목록이라 오탐이 없다. 정상 산출물에는 절대 나올 수 없는 문자열이다.
# ════════════════════════════════════════════════════════════════

_INTERNAL_MARKS = ("[[[", "]]]", "<BLANK_A>", "<BLANK_B>", "<CORE_BLANK>")


def internal_marker_leaks(data) -> list:
    """산출 데이터 안에 내부 마커가 남은 자리 목록 [(경로, 마커), ...].

    지문(paragraphs)은 마커를 담고 있는 게 정상이므로 제외한다.
    """
    SKIP = {"paragraphs", "paragraphs_render", "paragraphs_marked"}
    found = []

    def walk(node, path):
        if isinstance(node, str):
            for m in _INTERNAL_MARKS:
                if m in node:
                    found.append((path, m))
                    return
        elif isinstance(node, dict):
            for k, v in node.items():
                if k in SKIP:
                    continue
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(data, "")
    return found


class _IdxUnknown(Exception):
    """LLM 이 준 idx 를 신뢰할 수 없어 위치 기반 검사를 건너뛴다는 신호 (_s156)."""


def q4_conflict_unsatisfiable(paragraphs, statements_evidence) -> bool:
    """Q4 근거가 지문의 **모든 문장**을 덮으면 True (_s156).

    q4_conflicts_with_answer 는 '정답 자리를 Q4 근거 문장 밖에서 고르라'는 조건이다.
    그런데 짧은 지문에서는 그럴 자리가 아예 없다.
    실측(25년 고1 9월 23번): 지문 5문장 / Q4 진술 5개 → 다섯 문장이 다 근거다.
    어느 낱말을 골라도 걸리므로 조건이 아니라 함정이 된다. 어휘 9번, 바깥 3번을
    통째로 잡아먹고 지문이 빠졌다.
    → 덮이지 않은 문장이 하나도 없으면 이 검사를 아예 걸지 않는다.
    """
    evs = [ _key_words(e) for e in (statements_evidence or []) if e ]
    if not evs:
        return False
    free = 0
    for p in (paragraphs or []):
        try:
            toks = p[1].split()
        except Exception:
            continue
        for lo, hi in _sentence_bounds(toks):
            key = _key_words(" ".join(toks[lo:hi + 1]))
            if len(key) < 3:
                continue
            covered = any(
                len(key & ek) / max(1, min(len(key), len(ek))) >= 0.7 for ek in evs if ek)
            if not covered:
                free += 1
    return free == 0
