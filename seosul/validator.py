"""
seosul/validator.py
서술형 종합 세트 자동 검증 — LLM 생성물이 제약을 지켰는지 코드로 확인한다.
실패 항목은 generator의 재생성 루프로 되돌려 보낸다.

검증 대상
  SA 본문 빈칸 배열영작 : 보기 토큰 == 정답 토큰(어형변형/중복 옵션 반영), 빈칸 복원 시 원문 일치
  SC 요약문 빈칸 배열영작 : 보기 토큰 == 정답 토큰(변형/중복 불가, 다중집합 동일)
  SD 어법 틀린 곳 고치기   : 각 오류가 grammar_points 화이트리스트에 매칭 + 블랙리스트 회피
                            + 빈칸 문장 아님 + 서로 다른 문장 + 개수 범위
  SE 어휘 품사 변형 채우기 : 보기 원형 존재 + 실제 형태 변형(answer != base) + 빈칸 문장에 위치
                            ★ base_pos가 '동사'면 굴절(-s/-ed/-ing)도 허용, 그 외 품사는 굴절 금지
공통 : SA/SD/SE가 점유한 문장이 서로 겹치지 않음(누더기 방지)
"""
import re
from typing import List, Dict, Tuple

# ---- 문장 분리 (지문 분할 규칙의 약어 예외 처리 재사용) ----
_ABBR = {"Ms.", "Mr.", "Mrs.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
         "e.g.", "i.e.", "vs.", "etc."}

def split_sentences(text: str) -> List[str]:
    """. ! ? 뒤 공백으로 분리하되 약어/1글자 이니셜은 문장끝으로 보지 않음."""
    text = re.sub(r"\s+", " ", text.strip())
    out, buf, i = [], "", 0
    toks = text.split(" ")
    for idx, w in enumerate(toks):
        buf = (buf + " " + w).strip()
        if w and w[-1] in ".!?":
            # 약어 예외
            if w in _ABBR:
                continue
            # 1글자 이니셜 (J. K.)
            if re.fullmatch(r"[A-Z]\.", w):
                continue
            # 따옴표로 닫히는 인용 ("...despair.") 도 문장끝 허용
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out


# ---- 토큰화 ----
def tokenize(s: str) -> List[str]:
    """배열영작 비교용. 따옴표/구두점 제거, 소문자, 공백 분리."""
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"[^\w'\-]+", " ", s)   # 단어/아포스트로피/하이픈만
    return [t for t in s.lower().split() if t]


def _multiset(lst: List[str]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for x in lst:
        d[x] = d.get(x, 0) + 1
    return d


# ---- 간이 어간 (어형변형 허용 비교용) ----
def _stem(w: str) -> str:
    w = w.lower()
    for suf in ("ings", "ing", "ied", "ies", "ied", "ed", "es", "s", "'s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


# 불규칙 변형 정규화 (모든 형태 → 기본형). SA 어형변형 매칭용.
def _build_irr():
    groups = [
        ("be", "is", "are", "am", "was", "were", "been", "being"),
        ("have", "has", "had", "having"),
        ("do", "does", "did", "done", "doing"),
        ("make", "makes", "made", "making"),
        ("come", "comes", "came", "coming"),
        ("become", "becomes", "became", "becoming"),
        ("go", "goes", "went", "gone", "going"),
        ("take", "takes", "took", "taken", "taking"),
        ("give", "gives", "gave", "given", "giving"),
        ("find", "finds", "found", "finding"),
        ("see", "sees", "saw", "seen", "seeing"),
        ("know", "knows", "knew", "known", "knowing"),
        ("grow", "grows", "grew", "grown", "growing"),
        ("begin", "begins", "began", "begun", "beginning"),
        ("bring", "brings", "brought", "bringing"),
        ("build", "builds", "built", "building"),
        ("buy", "buys", "bought", "buying"),
        ("catch", "catches", "caught", "catching"),
        ("choose", "chooses", "chose", "chosen", "choosing"),
        ("draw", "draws", "drew", "drawn", "drawing"),
        ("drive", "drives", "drove", "driven", "driving"),
        ("eat", "eats", "ate", "eaten", "eating"),
        ("fall", "falls", "fell", "fallen", "falling"),
        ("feel", "feels", "felt", "feeling"),
        ("fight", "fights", "fought", "fighting"),
        ("get", "gets", "got", "gotten", "getting"),
        ("hold", "holds", "held", "holding"),
        ("keep", "keeps", "kept", "keeping"),
        ("lead", "leads", "led", "leading"),
        ("leave", "leaves", "left", "leaving"),
        ("lose", "loses", "lost", "losing"),
        ("mean", "means", "meant", "meaning"),
        ("meet", "meets", "met", "meeting"),
        ("pay", "pays", "paid", "paying"),
        ("rise", "rises", "rose", "risen", "rising"),
        ("run", "runs", "ran", "running"),
        ("say", "says", "said", "saying"),
        ("seek", "seeks", "sought", "seeking"),
        ("sell", "sells", "sold", "selling"),
        ("send", "sends", "sent", "sending"),
        ("show", "shows", "showed", "shown", "showing"),
        ("speak", "speaks", "spoke", "spoken", "speaking"),
        ("spend", "spends", "spent", "spending"),
        ("stand", "stands", "stood", "standing"),
        ("teach", "teaches", "taught", "teaching"),
        ("tell", "tells", "told", "telling"),
        ("think", "thinks", "thought", "thinking"),
        ("understand", "understands", "understood", "understanding"),
        ("win", "wins", "won", "winning"),
        ("write", "writes", "wrote", "written", "writing"),
        ("hear", "hears", "heard", "hearing"),
        ("read", "reads", "reading"),
        # 불규칙 복수명사 (접두 매칭이 안 되는 것들)
        ("man", "men"), ("woman", "women"), ("child", "children"),
        ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"),
        ("mouse", "mice"), ("person", "people"),
    ]
    d = {}
    for g in groups:
        base = g[0]
        for form in g:
            d[form] = base
    return d


_IRREGULAR = _build_irr()


# =========================================================
#  SA / SC : 배열영작 보기↔정답 토큰 대조
# =========================================================
def validate_arrangement(bogi: List[str], answers: Dict[str, str],
                         allow_inflect: bool, allow_dup: bool) -> List[str]:
    errs: List[str] = []
    bogi_tok = tokenize(" ".join(bogi))
    ans_tok = tokenize(" ".join(answers.values()))

    if not allow_inflect and not allow_dup:
        # 변형/중복 불가 → 다중집합 완전 동일 (SC)
        if _multiset(bogi_tok) != _multiset(ans_tok):
            extra = _multiset(ans_tok)
            for k, v in _multiset(bogi_tok).items():
                extra[k] = extra.get(k, 0) - v
            mism = {k: v for k, v in extra.items() if v != 0}
            errs.append(f"[보기≠정답] 다중집합 불일치: {mism}")
        return errs

    if allow_dup and not allow_inflect:
        # 중복 허용·변형 불가 → 정답의 모든 토큰이 보기에 존재 + 보기 전부 1회 이상 사용
        bset = set(bogi_tok)
        for t in ans_tok:
            if t not in bset:
                errs.append(f"[보기에 없는 단어] '{t}'")
        for t in bset:
            if t not in set(ans_tok):
                errs.append(f"[미사용 보기] '{t}'")
        return errs

    # 어형변형 허용 (SA) → 기능어는 보기에 그대로 존재, 내용어는 어근 접두/불규칙 매칭
    FUNC = {"a", "an", "the", "in", "on", "of", "to", "and", "or", "for", "with",
            "as", "that", "than", "by", "at", "from", "into", "about", "but", "so"}
    bset = set(bogi_tok)

    def _can(w: str) -> str:
        return _IRREGULAR.get(w, w)

    def _ves(w: str) -> set:
        """-f/-fe ↔ -ves 복수 정규화. life↔lives, leaf↔leaves 등."""
        if w.endswith("ves"):
            return {w, w[:-3] + "f", w[:-3] + "fe"}
        if w.endswith("fe"):
            return {w, w[:-2] + "ves"}
        if w.endswith("f"):
            return {w, w[:-1] + "ves"}
        return {w}

    def _matches(t: str) -> bool:
        if t in bset:
            return True
        ct = _can(t)
        tv = _ves(t)
        for b in bogi_tok:
            if ct == _can(b):                       # 불규칙 정규화 (made↔make)
                return True
            if tv & _ves(b):                        # f/ves 복수 (life↔lives, leaf↔leaves)
                return True
            if t[:3] == b[:3]:                      # 앞 3글자 공유 (illuminate↔illumination)
                return True
            if t.startswith(b[:max(3, len(b) - 2)]) or b.startswith(t[:max(3, len(t) - 2)]):
                return True
        return False

    for t in ans_tok:
        if t in FUNC:
            if t not in bset:
                errs.append(f"[보기에 없는 기능어] '{t}' (관사·전치사도 보기에 포함해야 함)")
        elif not _matches(t):
            errs.append(f"[보기에 없는 단어] '{t}'")
    return errs


def validate_reconstruction(sentence_tpl: str, label: str, answer: str,
                            original_sentence: str) -> List[str]:
    """SA: 빈칸에 정답을 넣으면 원문 문장과 일치하는가."""
    filled = sentence_tpl.replace("{{%s}}" % label, answer)
    if tokenize(filled) != tokenize(original_sentence):
        return [f"[복원 불일치] ({label}) 빈칸 복원 결과가 원문과 다름"]
    return []


# =========================================================
#  SD : 어법 오류 ↔ grammar_points 화이트리스트
# =========================================================
_BE_FORMS = {"is", "are", "was", "were", "am"}

def _agreement_pair(w, r):
    wl, rl = (w or "").lower(), (r or "").lower()
    if wl in _BE_FORMS and rl in _BE_FORMS:
        return True
    if wl == rl + "s" or rl == wl + "s" or {wl, rl} == {"has", "have"}:
        return True
    return False

def _adjacent_subject(sent_text, wrong):
    if not sent_text:
        return False
    m = re.search(rf"(\w+)\s+{re.escape(wrong)}(?![A-Za-z])", sent_text)
    if not m:
        return False
    prev = m.group(1).lower()
    return prev not in {"the", "a", "an", "of", "in", "on", "to", "that", "which",
                        "who", "and", "or", "but", "they", "we", "it", "he", "she"}

def _forbidden_keywords(gp_index):
    kws = ["관사", "a/an/the", "article", "어휘", "철자", "혼동",
           "affect", "effect", "둘 다 맞", "둘다 맞", "both correct",
           "병렬", "조동사", "사역", "지각", "관용", "전치사 관용", "스펠"]
    for g in (gp_index or {}).values():
        if "출제 금지" in (g.get("prohibited_analysis") or ""):
            nm = (g.get("name") or "").strip()
            if nm:
                kws.append(nm)
    return kws

def error_is_forbidden(e: dict, gp_index: Dict[int, dict], sentences: List[str] = None) -> str:
    """개별 어법오류가 '출제 금지'면 사유 문자열, 아니면 ''. 검증기·합성기 공용."""
    # ★ wrong/right 단어 자체도 검사 대상 — why를 '수동 분사'로 적어 우회하던 것 차단
    #   (예: unaffecting→unaffected 가 affect 금지어에 걸리게 된다)
    blob = (f"{e.get('category','')} {e.get('name','')} {e.get('why','')} "
            f"{e.get('wrong','')} {e.get('right','')}").lower()
    for kw in _forbidden_keywords(gp_index):
        if kw and kw.lower() in blob:
            return f"금지유형(키워드 '{kw}')"
    gp = (gp_index or {}).get(e.get("gp_id"))
    if gp and "출제 금지" in (gp.get("prohibited_analysis") or ""):
        return "출제 금지 gp"
    if sentences is not None and _agreement_pair(e.get("wrong"), e.get("right")):
        si = e.get("sent")
        stext = sentences[si] if isinstance(si, int) and 0 <= si < len(sentences) else ""
        if _adjacent_subject(stext, e.get("wrong", "")):
            return "근접 수일치(주어 바로 뒤)"
    return ""


def validate_grammar_errors(errors: List[dict], gp_index: Dict[int, dict],
                            blank_sentences: set, n_range=(4, 5),
                            single_passage: bool = False,
                            sentences: List[str] = None) -> List[str]:
    """
    errors: [{sent, wrong, right, category, why, (gp_id 선택)}]
    설계: 모델이 DB id를 정확히 맞히는 건 불안정하므로 강제하지 않는다.
    하드 게이트는 '블랙리스트(출제 금지 유형) 차단' + 구조 검사.
    화이트리스트는 프롬프트에서 모델을 '유도'하는 용도로만 쓴다.
    """
    errs: List[str] = []
    lo, hi = (2, 3) if single_passage else n_range
    if not (lo <= len(errors) <= hi):
        errs.append(f"[개수] 어법 오류 {len(errors)}개 (허용 {lo}~{hi})")

    seen_sent, seen_cat = set(), []
    for e in errors:
        fb = error_is_forbidden(e, gp_index, sentences)
        if fb:
            errs.append(f"[블랙리스트] '{e.get('wrong')}→{e.get('right')}' {fb}")
        if e.get("sent") in blank_sentences:
            errs.append(f"[겹침] 문장{e.get('sent')}은 빈칸 문장 → 어법 오류 금지")
        if e.get("sent") in seen_sent:
            errs.append(f"[중복문장] 문장{e.get('sent')}에 오류 2개")
        seen_sent.add(e.get("sent"))
        gp = (gp_index or {}).get(e.get("gp_id"))
        seen_cat.append(e.get("category") or (gp.get("category") if gp else None))
        if e.get("wrong") == e.get("right"):
            errs.append(f"[무변화] wrong==right ('{e.get('wrong')}')")

    for a, b in zip(seen_cat, seen_cat[1:]):
        if a and a == b:
            errs.append(f"[유형반복] 연속 동일 유형 '{a}'")
    return errs


# =========================================================
#  SE : 품사 변형 채우기
# =========================================================
def _is_inflection_only(base: str, ans: str, base_pos: str = "") -> bool:
    """단순 굴절(복수/3인칭 -s/-es, 불규칙 복수)인지 — 품사 변형이 아님.

    ★ base가 '동사'이면 굴절도 유효한 변형으로 인정한다.
      produce→produces(3인칭), prove→proved(과거), adhere→adhering(동명사) 모두 통과.
      base_pos가 비어 있거나 동사가 아니면 종전대로 굴절을 금지한다
      (photograph→photographs 같은 명사 복수 차단)."""
    b, a = base.lower(), ans.lower()
    if base_pos and "동사" in str(base_pos):
        return False
    if a in (b + "s", b + "es"):
        return True
    if b.endswith("y") and a == b[:-1] + "ies":      # study→studies
        return True
    if _IRREGULAR.get(a) == b and a != b:            # 불규칙 복수/시제만(man→men 등)
        # 단, 파생(동→명 등)이 아니라 같은 표제어의 굴절이면 변형으로 안 침
        return a.endswith("s") or a in ("men", "women", "children", "feet", "teeth", "mice", "people", "geese")
    return False

def validate_word_forms(blanks: List[dict], bogi: List[str],
                        blank_sentences: set, sentences: List[str] = None) -> List[str]:
    errs: List[str] = []
    bset = {b.lower() for b in bogi}
    for bl in blanks:
        base, ans = bl.get("base", "").lower(), bl.get("answer", "")
        al = ans.lower()
        if base not in bset:
            errs.append(f"[보기없음] ({bl.get('label')}) 원형 '{base}' 보기에 없음")
        if al == base:
            errs.append(f"[무변형] ({bl.get('label')}) '{ans}' 형태 변형 안 됨")
        elif _is_inflection_only(base, ans, bl.get("base_pos", "")):
            errs.append(f"[단순굴절] ({bl.get('label')}) '{base}→{ans}'는 굴절일 뿐 품사 변형 아님 "
                        f"(base_pos='{bl.get('base_pos','')}' — 원형이 동사면 base_pos에 '동사'라고 적어라)")
        root = base[:max(3, len(base) - 2)] if base else ""
        if root and not al.startswith(root):
            errs.append(f"[어근불일치] ({bl.get('label')}) '{ans}'가 원형 '{base}'에서 안 나옴")
        # ★ 정답이 실제 그 문장 원문에 존재해야 빈칸이 뚫린다(유령 빈칸 차단)
        if sentences is not None:
            si = bl.get("sent")
            if not (isinstance(si, int) and 0 <= si < len(sentences)
                    and re.search(rf"(?<![A-Za-z]){re.escape(ans)}(?![A-Za-z])", sentences[si])):
                errs.append(f"[원문부재] ({bl.get('label')}) 정답 '{ans}'가 문장[{si}] 원문에 없음 → 빈칸 안 뚫림")
        if bl.get("sent") not in blank_sentences:
            errs.append(f"[위치] ({bl.get('label')}) 빈칸 문장에 없음")
    return errs


# =========================================================
#  문장 역할 비겹침 (누더기 방지)
# =========================================================
def validate_role_overlap(roles: Dict[str, List[int]]) -> List[str]:
    errs, used = [], {}
    for typ, idxs in roles.items():
        for i in idxs:
            if i in used:
                errs.append(f"[역할겹침] 문장{i}: {used[i]} & {typ} 동시 점유")
            used[i] = typ
    return errs


# =========================================================
#  세트 전체 검증
# =========================================================
def validate_set(s: dict, gp_index: Dict[int, dict]) -> Tuple[bool, List[str]]:
    errs: List[str] = []
    roles = s.get("roles", {})
    errs += validate_role_overlap(roles)
    blank_sents = set(roles.get("SA", [])) | set(roles.get("SE", []))

    for item in s.get("items", []):
        t = item["type"]
        if t == "SA":
            errs += validate_arrangement(item["bogi"], item["answers"],
                                         allow_inflect=item.get("allow_inflect", True),
                                         allow_dup=item.get("allow_dup", True))
            # 두 빈칸이 서로 포함관계면 누더기 → 거부
            avals = [v for v in item.get("answers", {}).values() if v]
            for i in range(len(avals)):
                for j in range(len(avals)):
                    if i != j and avals[i] in avals[j]:
                        errs.append(f"[빈칸겹침] SA 정답이 서로 포함관계: '{avals[i]}' ⊂ '{avals[j]}'")
            for lab, meta in item.get("blanks", {}).items():
                if "tpl" in meta and "original" in meta:
                    errs += validate_reconstruction(meta["tpl"], lab,
                                                    item["answers"][lab], meta["original"])
        elif t == "SC":
            errs += validate_arrangement(item["bogi"], item["answers"],
                                         allow_inflect=False, allow_dup=False)
        elif t == "SD":
            errs += validate_grammar_errors(item["errors"], gp_index, blank_sents,
                                            single_passage=s.get("single_passage", True))
        elif t == "SE":
            errs += validate_word_forms(item["blanks"], item["bogi"], set(roles.get("SE", [])))

    return (len(errs) == 0), errs
