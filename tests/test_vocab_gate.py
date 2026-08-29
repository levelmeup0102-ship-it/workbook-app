# -*- coding: utf-8 -*-
"""_s142 관문 회귀 테스트 — 세 길이 전부 같은 목록을 보는지 확인한다.

이 테스트가 지키는 것: 목록에 단어를 넣었는데 어느 한 길만 막히는 상황.
그게 _s135('why?')·_s140('years' 'people') 의 원인이었다.
"""
import os, re, sys, ast
os.environ.setdefault("SUPABASE_URL", ""); os.environ.setdefault("ANTHROPIC_API_KEY", "x")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variation import vocab_q3 as V

fails = []
def check(cond, msg):
    if not cond: fails.append(msg)

# ── 1. 관문 자체 ─────────────────────────────────────────────
for w, tag in [("why", "기능어"), ("why?", "기능어(구두점 붙어도)"), ("The", "기능어"),
               ("however", "담화표지"), ("Similarly,", "담화표지"),
               ("years", "구체명사"), ("people", "구체명사"), ("audience", "구체명사"),
               ("of", "너무 짧음"), ("", "빈 값")]:
    check(V.blocked_reason(w) is not None, f"관문이 '{w}'({tag})를 안 막았다")

for w in ["significant", "modesty", "dissuade", "abandon", "rarely", "concern"]:
    check(V.blocked_reason(w) is None, f"관문이 정상 단어 '{w}'를 막았다")

# ── 2. 세 길이 전부 관문을 지나는가 ──────────────────────────
src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "variation", "vocab_q3.py"), encoding="utf-8").read()
tree = ast.parse(src)
LISTS = {"_VOCAB_STOP", "_CONCRETE", "_DISCOURSE_MARKER"}
for node in ast.walk(tree):
    # _key_words 는 밑줄 자격을 정하지 않는다 — 문장끼리 비교할 때 기능어를 걷어낼 뿐이라
    # 관문과 목적이 다르다 (_s147). 자격 판정 함수가 아니므로 예외.
    _EXEMPT = ("blocked_reason", "is_discourse_marker", "_key_words")
    if isinstance(node, ast.FunctionDef) and node.name not in _EXEMPT:
        used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & LISTS
        check(not used, f"{node.name}() 이 관문을 안 거치고 {sorted(used)} 를 직접 본다")

check("answer_pos_ok" not in {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)},
      "죽은 검사대 answer_pos_ok 가 아직 있다")

# ── 3·4. 지문·5항목을 실제 모양으로 갖춘 fixture ─────────────
#   normalize / validate 는 '5개인가' '한 문장에 몰리지 않았나' 를 먼저 보므로
#   서로 다른 문장에서 5자리를 잡고, 그중 한 자리에만 금지어를 넣는다.
PA = "Sleep matters a lot. This finding seems significant to many."
PB = "The researchers found that people gain years of insight however clearly today."
PC = "It makes the outcome durable and strong. Everyone involved stays engaged and alert."
paras = [("A", PA), ("B", PB), ("C", PC)]
T = {0: PA.split(), 1: PB.split(), 2: PC.split()}

# (para, 단어) — 전부 서로 다른 문장
SLOTS = [(0, "matters"), (0, "significant"), (1, "insight"), (2, "durable"), (2, "engaged")]
# 선지·반의어는 서로 어근이 겹치면 안 된다 (_s139·_s141)
SHOWN = ["counts", "trivial", "blindness", "fragile", "bored"]
ANTON = ["ignores", "minor", "clarity", "sturdy", "eager"]

def five(bad_word=None):
    """5항목 fixture. bad_word 가 있으면 3번 자리(para B)를 그 단어로 바꾼다."""
    items = []
    for k, (pi, w) in enumerate(SLOTS):
        items.append({"n": k + 1, "para": pi, "idx": T[pi].index(w),
                      "original": w, "shown": SHOWN[k],
                      "antonym": ANTON[k], "is_answer": (k == 2)})
    if bad_word:
        items[2].update(para=1, idx=T[1].index(bad_word), original=bad_word)
    return items

BAD = ["people", "years", "however", "clearly"]

# ── 3. LLM 픽 경로(평소 쓰는 길) ─────────────────────────────
check(V.normalize_llm_vocab(five(), paras) is not None,
      "정상 fixture 가 LLM 픽 경로에서 거부됐다 — 테스트가 무의미해진다")
for bad in BAD:
    check(V.normalize_llm_vocab(five(bad), paras) is None,
          f"LLM 픽 경로가 '{bad}' 를 통과시켰다")

# ── 4. 마지막 확인(validate_vocab) ───────────────────────────
check(not V.validate_vocab(five(), paras, pid="테스트"),
      "정상 fixture 를 validate_vocab 이 거부했다 — 테스트가 무의미해진다")
for bad in BAD:
    errs = V.validate_vocab(five(bad), paras, pid="테스트")
    check(any(bad in e for e in errs), f"validate_vocab 이 '{bad}' 를 안 잡았다")

# ── 5. 코드 픽(폴백)도 막는가 ────────────────────────────────
cands = V.vocab_candidates(paras)
bare = {c["bare"] for c in cands}
for bad in ["people", "years", "however", "clearly"]:
    check(bad not in bare, f"코드 픽이 '{bad}' 를 후보로 냈다")


# ── 6. 부정관사 a/an 보정 (_s143) ────────────────────────────
#   실측: 'an enclosed space' 의 enclosed 를 sealed 로 바꾸자 'an sealed' 가 됐다.
apara = [("A", "An echo chamber refers to an enclosed space where sound stays.")]
aitem = {"n": 1, "para": 0, "idx": 6, "original": "enclosed", "shown": "sealed",
         "antonym": "open", "is_answer": True}
rendered = re.sub(r"<[^>]+>", "", V.apply_vocab_items(apara, [aitem])[0][1])
check("to a " in rendered and "an sealed" not in rendered,
      f"렌더가 관사를 안 고쳤다: {rendered}")

# 소리 규칙이라 코드가 다 못 맞힌다 — 확실한 것만 고치고 애매하면 그대로 둔다
for prev, shown, want in [("an", "sealed", "a"), ("a", "enclosed", "an"),
                          ("an", "hour", "an"), ("a", "university", "a"),
                          ("an", "honest", "an"), ("An", "sealed", "A"),
                          ("the", "sealed", "the"), ("a", "FBI", "a")]:
    got = V._fix_article_token(prev, shown)
    check(got == want, f"관사 보정 '{prev} {shown}' → '{got}' (기대 '{want}')")

# 렌더에만 있으면 옛 캐시를 못 잡는다 — validate_vocab 도 본다
bad5 = [dict(aitem, n=k + 1, is_answer=(k == 0)) for k in range(5)]
check(any("부정관사" in e for e in V.validate_vocab(bad5, apara, pid="T")),
      "validate_vocab 이 관사 불일치를 안 잡았다")


# ── 7. 구두점 (_s144) ────────────────────────────────────────
import variation.generator as G

# (1) 약어 마침표는 문장 경계가 아니다 — 'to 7 p.m.' 이 'to 7 p' 로 잘리면 안 된다
for text, want in [
    ("extend the library's operating hours to 7 p.m.", "extend the library's operating hours to 7 p.m."),
    ("extend the library's operating hours to 7 p.m. This change would", "extend the library's operating hours to 7 p.m."),
    ("earned his Ph.D. from Cornell University in 1961", "earned his Ph.D. from Cornell University in 1961"),
    ("studied at Cornell University in 1961. In 1967", "studied at Cornell University in 1961"),
    ("convince more readers for the whole story.", "convince more readers for the whole story"),
]:
    got = G._cut_before_punct(text, 4, sentence_only=True)
    check(got == want, f"빈칸 절단 '{text[:40]}…' → '{got}' (기대 '{want}')")

# (2) 밑줄 밖으로 구두점 — 원문 문장 끝 마침표가 사라지면 안 된다
_t = "he showed that whales use them to communicate."
_r = V.apply_vocab_items([("A", _t)], [{"n": 1, "para": 0, "idx": 7,
      "original": "communicate.", "shown": "interact", "antonym": "x", "is_answer": True}])[0][1]
check(_r.endswith("</u>."), f"문장 끝 마침표가 밑줄 밖에 안 붙었다: {_r[-40:]}")
check("<u class=\"vword\">interact</u>" in _r, f"밑줄 안에 구두점이 남았다: {_r[-40:]}")

# (3) 주격 전용 대명사로 끝나면 거부, you/it 은 허용
for ph, want_ok in [("got a little rush of excitement, knowing I", False),
                    ("said that after the meeting he", False),
                    ("the real product being sold is you", True),
                    ("makes the whole thing work for it", True)]:
    ok = G._clean_boundary_ok(ph, ph + " was there", strict=False)
    check(ok == want_ok, f"경계 판정 '{ph}' → {'허용' if ok else '거부'} (기대 {'허용' if want_ok else '거부'})")


# ── 8. 구동사 불변화사 (_s146) ───────────────────────────────
_pt = "These bacteria can break down plastic quickly and safely every single day"
_tk = _pt.split(); _ix = _tk.index("break")
_items = [dict(n=k+1, para=0, idx=_ix if k == 0 else 6+k,
               original=_tk[_ix] if k == 0 else _tk[6+k],
               shown="decompose" if k == 0 else f"z{k}word",
               antonym=f"a{k}", is_answer=(k == 0)) for k in range(5)]
check(any("구동사" in e for e in V.validate_vocab(_items, [("A", _pt)], pid="T")),
      "validate_vocab 이 'decompose down' 을 안 잡았다")
check(V.particle_follows("down") and V.particle_follows("up.") and not V.particle_follows("the"),
      "particle_follows 판정이 이상하다")
# 코드 픽도 구동사 자리를 안 낸다
_c = V.vocab_candidates([("A", "Intro sentence here for padding. " + _pt + ".")])
check(all(x["bare"] != "break" for x in _c), "코드 픽이 구동사 자리를 후보로 냈다")


# ── 9. 문항끼리 답 흘리기 (_s148) ────────────────────────────
_st = [("가", "Roger Payne earned his Ph.D. from Harvard University.", False),
       ("나", "Payne discovered whale songs in 1967.", True),
       ("다", "The album was a commercial failure.", False),
       ("라", "Ocean Alliance was founded to protect whales and the oceans.", True),
       ("마", "The global ban began in 1986.", True)]
check(V.statements_leak_blanks(_st, "founded Ocean Alliance to protect whales and the earth's oceans",
                               "led more than 100 research trips worldwide") == [("라", "A")],
      "Q5 정답 노출 진술을 못 잡았다")
check(V.statements_leak_blanks(_st, "were facing a severe lack of funds",
                               "provided cough plates from their patients") == [],
      "노출이 없는데 잡았다 — 오탐")

_bp = [("A", "They found the answer. To hinder them, local doctors and nurses <BLANK_B>. The patients coughed.")]
_bt = _bp[0][1].split()
for _w, _want in [("hinder", True), ("found", False), ("coughed.", False)]:
    _it = [dict(n=1, para=0, idx=_bt.index(_w), original=_w, shown="x",
                antonym="y", is_answer=True)]
    check(V.answer_in_blank_sentence(_it, _bp) == _want,
          f"정답 '{_w}' 의 빈칸 문장 판정이 틀렸다")

# ── 10. 전치사·불변화사 비문 (_s175) ────────────────────────
#   오늘 하루에 다섯 자리가 비문인 채로 인쇄됐다. 세 길 전부에서 막혀야 한다.
_BAD = [
    # (지문, 원문 낱말, LLM 이 낸 낱말, 무엇에 걸려야 하는가)
    ("Eventually, a lack of sleep catches up with you and the debt comes due.",
     "catches", "overtakes", "구동사"),
    ("A symphony isn't the goal. Leave the conductor and the sheet music behind.",
     "Leave", "Discard", "불변화사"),
    ("Literature throughout history has dreamed of creating human-like machines.",
     "dreamed", "envisioned", "지배하는"),
    ("In time they became capable of spreading out from Africa to the world.",
     "capable", "able", "지배하는"),
    ("No mammal gave those peoples more opportunity to domesticate than gazelles.",
     "opportunity", "impediment", "지배하는"),
]
for _txt, _orig, _shown, _why in _BAD:
    _tk = _txt.split()
    _ix = next(k for k, w in enumerate(_tk) if V.strip_edge_punct(w).lower() == _orig.lower())
    # (1) 코드 픽 후보에서 빠져야 한다
    check(all(c["idx"] != _ix for c in V.vocab_candidates([["A", _txt]])),
          f"코드 픽이 '{_orig}' 자리를 후보로 냈다 ({_why})")
    # (2) 백스톱(validate_vocab)이 막아야 한다
    _it = [dict(n=1, para=0, idx=_ix, original=_tk[_ix], shown=_shown,
                antonym=_shown, is_answer=True)]
    _errs = " ".join(V.validate_vocab(_it * 5, [["A", _txt]], pid="t"))
    check(_why in _errs, f"백스톱이 '{_orig} → {_shown}' 를 안 막았다 ({_why})")

# 헛걸림 — 동의어가 전치사를 그대로 받는 자리는 막지 않는다
for _cur, _nx in [("significant", "to"), ("important", "to"), ("meaning", "of"),
                  ("majority", "of"), ("optimistic", "about"), ("persistence", "of")]:
    check(V.governed_prep(_cur, _nx) == "",
          f"지배어 표가 정상 자리 '{_cur} {_nx}' 를 막았다")
# 앞말이 문장을 끝내면 뒤 전치사는 다음 문장 것이다
check(V.governed_prep("dreams.", "of") == "", "문장 경계를 못 봤다")
# 부사는 불변화사를 지배하지 않는다 / 절이 갈리면 걸음을 멈춘다
check(V.split_particle_after("You gradually slow down.".split(), 1) == "",
      "부사 자리를 불변화사 자리로 봤다")
check(V.split_particle_after("trying to herd an animal that runs away,".split(), 0) == "",
      "절이 갈렸는데 계속 걸어갔다")
check(V.split_particle_after("spreading out from Africa, eventually".split(), 0) == "",
      "뒤에 목적어가 있는 전치사를 불변화사로 봤다")


# ── 11. para·idx 없이 자리 잡기 (_s179) ────────────────────
#   모델은 낱말과 앞 세 낱말만 적고, 자리는 코드가 찾는다.
_P = [["(A)", "Assimilation refers to the process of including new information into "
               "existing schemas or what we already know."],
      ["(B)", "In accommodation, existing schemas are changed in response to new "
               "situations and experiences."],
      ["(C)", "But when he sees a cow, he will need to modify the schema, as the new "
               "schema will not fit into the existing schema of a dog."]]
def _one(orig, before):
    _it = dict(n=1, original=orig, before=before, antonym="opposite",
               shown=orig + "x", is_answer=True)
    _o = V.normalize_llm_vocab([_it] * 5, _P, pid="t", report=[])
    return (_o[0]["para"], _o[0]["idx"]) if _o else None

# 지문에 한 번만 나오는 낱말 — before 가 없어도 잡힌다
check(_one("refers", "") == (0, 1), "유일한 낱말의 자리를 못 잡았다")
# 같은 낱말이 네 번 — before 로 갈린다
_tk0, _tk1, _tk2 = (p[1].split() for p in _P)
check(_one("existing", "new information into") == (0, _tk0.index("existing")),
      "(A)의 existing 을 못 가렸다")
check(_one("existing", "In accommodation,") == (1, _tk1.index("existing")),
      "(B)의 existing 을 못 가렸다")
check(_one("existing", "fit into the") == (2, len(_tk2) - 1 - _tk2[::-1].index("existing")),
      "(C)의 existing 을 못 가렸다")
# 모델이 para·idx 를 아예 안 줘도 된다
check(_one("modify", "need to") == (2, _tk2.index("modify")), "before 만으로 못 잡았다")
# 지문에 없는 낱말은 사유를 돌려준다
_rep = []
V.normalize_llm_vocab([dict(n=1, original="zzzz", before="", antonym="a",
                            shown="b", is_answer=True)] * 5, _P, pid="t", report=_rep)
check(any("지문에 없다" in x for x in _rep), "지문에 없는 낱말을 안 걸렀다")


# 약어 마침표를 문장 끝으로 보지 않는다
check(len(V._sentence_bounds("It closes at 5 p.m. every day today.".split())) == 1,
      "약어 마침표를 문장 끝으로 봤다")

print("\n".join("❌ " + f for f in fails) if fails else "✅ 전부 통과")
sys.exit(1 if fails else 0)
