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

    # 어형변형 허용 (SA) → 기능어는 보기에 그대로 존재, 내용어는 어근 접두 매칭
    FUNC = {"a", "an", "the", "in", "on", "of", "to", "and", "or", "for", "with",
            "as", "that", "than", "by", "at", "from", "into", "about", "but", "so"}
    bset = set(bogi_tok)

    def _matches(t: str) -> bool:
        if t in bset:
            return True
        for b in bogi_tok:
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
def validate_grammar_errors(errors: List[dict], gp_index: Dict[int, dict],
                            blank_sentences: set, n_range=(4, 5),
                            single_passage: bool = False) -> List[str]:
    """
    errors: [{sent, wrong, right, gp_id, why}]
    gp_index: {id: grammar_points row dict}  (active, prohibited_analysis 포함)
    blank_sentences: 빈칸이 들어간 문장 인덱스 집합 (오류 금지)
    """
    errs: List[str] = []
    lo, hi = (2, 3) if single_passage else n_range  # 단일지문이면 자리부족 → 2~3
    if not (lo <= len(errors) <= hi):
        errs.append(f"[개수] 어법 오류 {len(errors)}개 (허용 {lo}~{hi})")

    seen_sent, seen_cat = set(), []
    for e in errors:
        gid = e.get("gp_id")
        gp = gp_index.get(gid)
        if gp is None:
            errs.append(f"[근거없음] gp_id={gid} 화이트리스트에 없음 → 즉석 출제 금지")
            continue
        if not gp.get("active", True):
            errs.append(f"[비활성] gp_id={gid} 비활성 항목")
        if (gp.get("prohibited_analysis") or "").strip() and "출제 금지" in gp["prohibited_analysis"]:
            errs.append(f"[블랙리스트] gp_id={gid}({gp.get('name')}) 출제 금지 규칙 위반")
        if e["sent"] in blank_sentences:
            errs.append(f"[겹침] 문장{e['sent']}은 빈칸 문장 → 어법 오류 금지")
        if e["sent"] in seen_sent:
            errs.append(f"[중복문장] 문장{e['sent']}에 오류 2개")
        seen_sent.add(e["sent"])
        seen_cat.append(gp.get("category"))
        if e.get("wrong") == e.get("right"):
            errs.append(f"[무변화] wrong==right ('{e.get('wrong')}')")

    # 같은 유형 연속(2개 연속 동일 category) 회피 권고
    for a, b in zip(seen_cat, seen_cat[1:]):
        if a and a == b:
            errs.append(f"[유형반복] 연속 동일 category '{a}'")
    return errs


# =========================================================
#  SE : 품사 변형 채우기
# =========================================================
def validate_word_forms(blanks: List[dict], bogi: List[str],
                        blank_sentences: set) -> List[str]:
    errs: List[str] = []
    bset = {b.lower() for b in bogi}
    for bl in blanks:
        base, ans = bl.get("base", "").lower(), bl.get("answer", "")
        if base not in bset:
            errs.append(f"[보기없음] ({bl.get('label')}) 원형 '{base}' 보기에 없음")
        if ans.lower() == base:
            errs.append(f"[무변형] ({bl.get('label')}) '{ans}' 형태 변형 안 됨")
        root = base[:max(3, len(base) - 2)] if base else ""
        if root and not ans.lower().startswith(root):
            errs.append(f"[어근불일치] ({bl.get('label')}) '{ans}'가 원형 '{base}'에서 안 나옴")
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
